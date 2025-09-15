use std::{
    collections::{BTreeSet, HashMap, HashSet},
    hash::Hash,
    iter,
    ops::RangeTo,
};

use ff::{Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use rand_core::{CryptoRng, RngCore};

use super::{
    circuit::{
        sealed::{self},
        Advice, Any, Assignment, Challenge, Circuit, Column, ConstraintSystem, Fixed, FloorPlanner,
        Instance, Selector,
    },
    lookup, permutation, vanishing, Error, ProvingKey,
};
#[cfg(feature = "committed-instances")]
use crate::poly::EvaluationDomain;
use crate::{
    circuit::Value,
    plonk::{traces::ProverTrace, trash},
    poly::{
        batch_invert_rational, commitment::PolynomialCommitmentScheme, Coeff,
        ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, PolynomialRepresentation, ProverQuery,
    },
    transcript::{Hashable, Sampleable, Transcript},
    utils::{arithmetic::eval_polynomial, rational::Rational},
};

#[cfg(feature = "committed-instances")]
/// Commit to a vector of raw instances. This function can be used to prepare
/// the committed instances that the verifier will be provided with when this
/// feature is enabled.
pub fn commit_to_instances<F, CS: PolynomialCommitmentScheme<F>>(
    params: &CS::Parameters,
    domain: &EvaluationDomain<F>,
    instances: &[F],
) -> CS::Commitment
where
    F: WithSmallOrderMulGroup<3> + Ord + FromUniformBytes<64>,
{
    let mut poly = domain.empty_lagrange();
    for (poly_eval, value) in poly.iter_mut().zip(instances.iter()) {
        *poly_eval = *value;
    }
    //CS::commit_lagrange(params, &poly)

    let mut poly_gpu = crate::DeviceMemPool::allocate::<F>(poly.len()); 
    crate::DeviceMemPool::mem_copy_htod(&mut poly_gpu, &poly.values);     
    let commit = CS::commit_lagrang_gpu(params, &poly_gpu);
    crate::DeviceMemPool::deallocate(poly_gpu);
    commit
}

/// This computes a proof trace for the provided `circuits` when given the
/// public parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
///
/// The trace can then be used to finalise proofs, or to fold them.
pub(crate) fn compute_trace<
    F,
    CS: PolynomialCommitmentScheme<F>,
    T: Transcript,
    ConcreteCircuit: Circuit<F>,
>(
    params: &CS::Parameters,
    pk: &ProvingKey<F, CS>,
    circuits: &[ConcreteCircuit],
    // The prover needs to get all instances in non-committed form. However,
    // the first `nb_committed_instances` instance columns are dedicated for
    // instances that the verifier receives in committed form.
    #[cfg(feature = "committed-instances")] nb_committed_instances: usize,
    instances: &[&[&[F]]],
    mut rng: impl RngCore + CryptoRng,
    transcript: &mut T,
) -> Result<ProverTrace<F>, Error>
where
    CS::Commitment: Hashable<T::Hash>,
    F: WithSmallOrderMulGroup<3>
        + Sampleable<T::Hash>
        + Hashable<T::Hash>
        + Hash
        + Ord
        + FromUniformBytes<64>,
{
    #[cfg(not(feature = "committed-instances"))]
    let nb_committed_instances: usize = 0;

    if circuits.len() != instances.len() {
        return Err(Error::InvalidInstances);
    }

    for instances in instances.iter() {
        if instances.len() != pk.vk.cs.num_instance_columns
            || instances.len() < nb_committed_instances
        {
            return Err(Error::InvalidInstances);
        }
    }

    // Hash verification key into transcript
    pk.vk.hash_into(transcript)?;

    let domain = &pk.vk.domain;
    let mut meta = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut meta, circuits[0].params());
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut meta);

    // Selector optimizations cannot be applied here; use the ConstraintSystem
    // from the verification key.
    let meta = &pk.vk.cs;

    struct InstanceSingle<F: PrimeField> {
        pub instance_values: Vec<Polynomial<F, LagrangeCoeff>>,
        pub instance_polys: Vec<Polynomial<F, Coeff>>,
    }

    let instance: Vec<InstanceSingle<F>> = instances
        .iter()
        .map(|instance| -> Result<InstanceSingle<F>, Error> {
            let instance_values = instance
                .iter()
                .enumerate()
                .map(|(i, values)| {
                    // Committed instances go first.
                    let is_committed_instance = i < nb_committed_instances;
                    let mut poly = domain.empty_lagrange();
                    assert_eq!(poly.len(), domain.n as usize);
                    if values.len() > (poly.len() - (meta.blinding_factors() + 1)) {
                        return Err(Error::InstanceTooLarge);
                    }
                    if !is_committed_instance {
                        transcript.common(&F::from_u128(values.len() as u128))?;
                    }

                    for (poly_eval, value) in poly.iter_mut().zip(values.iter()) {
                        if !is_committed_instance {
                            transcript.common(value)?;
                        }
                        *poly_eval = *value;
                    }

                    if is_committed_instance {
                        //transcript.common(&CS::commit_lagrange(params, &poly))?;

                        let mut poly_gpu = crate::DeviceMemPool::allocate::<F>(poly.len()); 
                        crate::DeviceMemPool::mem_copy_htod(&mut poly_gpu, &poly.values);     
                        transcript.common(&CS::commit_lagrang_gpu(params, &poly_gpu))?;
                        crate::DeviceMemPool::deallocate(poly_gpu);                        
                    }

                    Ok(poly)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let instance_polys: Vec<_> = instance_values
                .iter()
                .map(|poly| {
                    //let lagrange_vec = domain.lagrange_from_vec(poly.to_vec());
                    //domain.lagrange_to_coeff(lagrange_vec);
                    let mut lagrange_vec = domain.empty_coeff();
                    let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(poly.values.len()); 
                    crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &poly.values);
                    crate::gpu_lagrange_to_coeff::<F>(&mut gpu_poly);        
                    crate::DeviceMemPool::mem_copy_dtoh(&mut lagrange_vec.values, &gpu_poly); 
                    crate::DeviceMemPool::deallocate(gpu_poly);
                    lagrange_vec
                })
                .collect();

            Ok(InstanceSingle {
                instance_values,
                instance_polys,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    #[derive(Clone)]
    struct AdviceSingle<F: PrimeField, B: PolynomialRepresentation> {
        pub advice_polys: Vec<Polynomial<F, B>>,
    }

    struct WitnessCollection<'a, F: Field> {
        k: u32,
        current_phase: sealed::Phase,
        advice: Vec<Polynomial<Rational<F>, LagrangeCoeff>>,
        unblinded_advice: HashSet<usize>,
        challenges: &'a HashMap<usize, F>,
        instances: &'a [&'a [F]],
        usable_rows: RangeTo<usize>,
        _marker: std::marker::PhantomData<F>,
    }

    impl<F: Field> Assignment<F> for WitnessCollection<'_, F> {
        fn enter_region<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about regions in this context.
        }

        fn exit_region(&mut self) {
            // Do nothing; we don't care about regions in this context.
        }

        fn enable_selector<A, AR>(&mut self, _: A, _: &Selector, _: usize) -> Result<(), Error>
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // We only care about advice columns here

            Ok(())
        }

        fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // Do nothing
        }

        fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.k));
            }

            self.instances
                .get(column.index())
                .and_then(|column| column.get(row))
                .map(|v| Value::known(*v))
                .ok_or(Error::BoundsFailure)
        }

        fn assign_advice<V, VR, A, AR>(
            &mut self,
            _: A,
            column: Column<Advice>,
            row: usize,
            to: V,
        ) -> Result<(), Error>
        where
            V: FnOnce() -> Value<VR>,
            VR: Into<Rational<F>>,
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // Ignore assignment of advice column in different phase than current one.
            if self.current_phase != column.column_type().phase {
                return Ok(());
            }

            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.k));
            }

            *self
                .advice
                .get_mut(column.index())
                .and_then(|v| v.get_mut(row))
                .ok_or(Error::BoundsFailure)? = to().into_field().assign()?;

            Ok(())
        }

        fn assign_fixed<V, VR, A, AR>(
            &mut self,
            _: A,
            _: Column<Fixed>,
            _: usize,
            _: V,
        ) -> Result<(), Error>
        where
            V: FnOnce() -> Value<VR>,
            VR: Into<Rational<F>>,
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // We only care about advice columns here

            Ok(())
        }

        fn copy(
            &mut self,
            _: Column<Any>,
            _: usize,
            _: Column<Any>,
            _: usize,
        ) -> Result<(), Error> {
            // We only care about advice columns here

            Ok(())
        }

        fn fill_from_row(
            &mut self,
            _: Column<Fixed>,
            _: usize,
            _: Value<Rational<F>>,
        ) -> Result<(), Error> {
            Ok(())
        }

        fn get_challenge(&self, challenge: Challenge) -> Value<F> {
            self.challenges
                .get(&challenge.index())
                .cloned()
                .map(Value::known)
                .unwrap_or_else(Value::unknown)
        }

        fn push_namespace<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about namespaces in this context.
        }

        fn pop_namespace(&mut self, _: Option<String>) {
            // Do nothing; we don't care about namespaces in this context.
        }
    }

    let (advice, challenges) = {
        let mut advice = vec![
            AdviceSingle::<F, LagrangeCoeff> {
                advice_polys: vec![domain.empty_lagrange(); meta.num_advice_columns],
            };
            instances.len()
        ];
        let mut challenges = HashMap::<usize, F>::with_capacity(meta.num_challenges);

        let unusable_rows_start = domain.n as usize - (meta.blinding_factors() + 1);
        for current_phase in pk.vk.cs.phases() {
            let column_indices = meta
                .advice_column_phase
                .iter()
                .enumerate()
                .filter_map(|(column_index, phase)| {
                    if current_phase == *phase {
                        Some(column_index)
                    } else {
                        None
                    }
                })
                .collect::<BTreeSet<_>>();

            for ((circuit, advice), instances) in
                circuits.iter().zip(advice.iter_mut()).zip(instances)
            {
                let mut witness = WitnessCollection {
                    k: domain.k(),
                    current_phase,
                    advice: vec![domain.empty_lagrange_rational(); meta.num_advice_columns],
                    unblinded_advice: HashSet::from_iter(meta.unblinded_advice_columns.clone()),
                    instances,
                    challenges: &challenges,
                    // The prover will not be allowed to assign values to advice
                    // cells that exist within inactive rows, which include some
                    // number of blinding factors and an extra row for use in the
                    // permutation argument.
                    usable_rows: ..unusable_rows_start,
                    _marker: std::marker::PhantomData,
                };

                // Synthesize the circuit to obtain the witness and other information.
                println!("synthesizing circuit");
                ConcreteCircuit::FloorPlanner::synthesize(
                    &mut witness,
                    circuit,
                    config.clone(),
                    meta.constants.clone(),
                )?;
                // println!("witness advice len: {}", witness.advice.len());
                // for (i, advice_col) in witness.advice.iter().enumerate() {
                //     println!(
                //         "witness advice col {}: {:?}",
                //         i,
                //         advice_col.values.iter().take(10).collect::<Vec<_>>()
                //     );
                // }
                // println!("witness advice: {:#?}", witness.advice);
                // ConcreteCircuit doesn't necessarily implement Debug
                // println!("circuit: {:?}", circuit);

                //let mut advice_values = batch_invert_rational::<F>(
                //    witness
                //        .advice
                //        .into_iter()
                //        .enumerate()
                //        .filter_map(|(column_index, advice)| {
                //            if column_indices.contains(&column_index) {
                //                Some(advice)
                //            } else {
                //                None
                //            }
                //        })
                //        .collect(),
                //);

                let mut advice_values = witness
                    .advice
                    .into_iter()
                    .enumerate()
                    .filter_map(|(column_index, advice)| {
                        if column_indices.contains(&column_index) {
                            Some(advice)
                        } else {
                            None
                        }
                    })
                    .map(|poly| {
                        let values = poly.values.into_iter().map(|a| a.numerator()).collect();
                        Polynomial {
                            values,
                            _marker: poly._marker,
                        }
                    })
                    .collect::<Vec<_>>();

                for (column_index, advice_values) in column_indices.iter().zip(&mut advice_values) {
                    if !witness.unblinded_advice.contains(column_index) {
                        for cell in &mut advice_values[unusable_rows_start..] {
                            *cell = F::random(&mut rng);
                        }
                    } else {
                        #[cfg(feature = "sanity-checks")]
                        for cell in &advice_values[unusable_rows_start..] {
                            assert_eq!(*cell, F::ZERO);
                        }
                    }
                }

                let advice_commitments: Vec<_> = advice_values
                    .iter()
                    //.map(|poly| CS::commit_lagrange(params, poly))
                    .map(|poly| {
                        let mut poly_gpu = crate::DeviceMemPool::allocate::<F>(poly.len()); 
                        crate::DeviceMemPool::mem_copy_htod(&mut poly_gpu, &poly.values);     
                        let commit = CS::commit_lagrang_gpu(params, &poly_gpu);
                        crate::DeviceMemPool::deallocate(poly_gpu);
                        commit
                    })
                    .collect();

                for commitment in &advice_commitments {
                    transcript.write(commitment)?;
                }
                for (column_index, advice_values) in column_indices.iter().zip(advice_values) {
                    advice.advice_polys[*column_index] = advice_values;
                }
            }

            for (index, phase) in meta.challenge_phase.iter().enumerate() {
                if current_phase == *phase {
                    let existing = challenges.insert(index, transcript.squeeze_challenge());
                    assert!(existing.is_none());
                }
            }
        }

        assert_eq!(challenges.len(), meta.num_challenges);
        let challenges = (0..meta.num_challenges)
            .map(|index| challenges.remove(&index).unwrap())
            .collect::<Vec<_>>();

        (advice, challenges)
    };

    // Sample theta challenge for keeping lookup columns linearly independent
    let theta: F = transcript.squeeze_challenge();

    let lookups: Vec<Vec<lookup::prover::Permuted<F>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Result<Vec<_>, Error> {
            // Construct and commit to permuted values for each lookup
            pk.vk
                .cs
                .lookups
                .iter()
                .map(|lookup| {
                    lookup.commit_permuted(
                        pk,
                        params,
                        domain,
                        theta,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                        &mut rng,
                        transcript,
                    )
                })
                .collect()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Sample beta challenge
    let beta: F = transcript.squeeze_challenge();

    // Sample gamma challenge
    let gamma: F = transcript.squeeze_challenge();

    // Commit to permutations.
    let permutations: Vec<permutation::prover::Committed<F>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| {
            pk.vk.cs.permutation.commit(
                params,
                pk,
                &pk.permutation,
                &advice.advice_polys,
                &pk.fixed_values,
                &instance.instance_values,
                beta,
                gamma,
                &mut rng,
                transcript,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let lookups: Vec<Vec<lookup::prover::Committed<F>>> = lookups
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            // Construct and commit to products for each lookup
            lookups
                .into_iter()
                .map(|lookup| lookup.commit_product(pk, params, beta, gamma, &mut rng, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Trash argument
    let trash_challenge: F = transcript.squeeze_challenge();

    let trashcans: Vec<Vec<trash::prover::Committed<F>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Result<Vec<_>, Error> {
            pk.vk
                .cs
                .trashcans
                .iter()
                .map(|trash| {
                    trash.commit::<CS, _>(
                        params,
                        domain,
                        trash_challenge,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                        transcript,
                    )
                })
                .collect()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Commit to the vanishing argument's random polynomial for blinding h(x_3)
    let vanishing = vanishing::Argument::<F, CS>::commit(params, domain, &mut rng, transcript)?;

    // Obtain challenge for keeping all separate gates linearly independent
    let y: F = transcript.squeeze_challenge();

    let (instance_polys, instance_values) = instance
        .into_iter()
        .map(|i| (i.instance_polys, i.instance_values))
        .unzip();

    Ok(ProverTrace {
        advice_polys: advice
            .into_iter()
            .map(|a| {
                a.advice_polys
                    .into_iter()
                    //.map(|p| domain.lagrange_to_coeff(p))
                    .map(|p|
                    { 
                        let mut lagrange_vec = domain.empty_coeff();
                        let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(p.values.len()); 
                        crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &p.values);
                        crate::gpu_lagrange_to_coeff::<F>(&mut gpu_poly);        
                        crate::DeviceMemPool::mem_copy_dtoh(&mut lagrange_vec.values, &gpu_poly); 
                        crate::DeviceMemPool::deallocate(gpu_poly);
                        lagrange_vec
                    })
                    .collect()
            })
            .collect(),
        instance_polys,
        instance_values,
        vanishing,
        lookups,
        trashcans,
        permutations,
        challenges,
        beta,
        gamma,
        theta,
        trash_challenge,
        y,
    })
}

/// This takes the computed trace of a set of witnesses and creates a proof
/// for the provided `circuit` when given the public
/// parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
pub(crate) fn finalise_proof<'a, F, CS: PolynomialCommitmentScheme<F>, T: Transcript>(
    params: &'a CS::Parameters,
    pk: &'a ProvingKey<F, CS>,
    // The prover needs to get all instances in non-committed form. However,
    // the first `nb_committed_instances` instance columns are dedicated for
    // instances that the verifier receives in committed form.
    #[cfg(feature = "committed-instances")] nb_committed_instances: usize,
    trace: ProverTrace<F>,
    transcript: &mut T,
) -> Result<(), Error>
where
    CS::Commitment: Hashable<T::Hash>,
    F: WithSmallOrderMulGroup<3>
        + Sampleable<T::Hash>
        + Hashable<T::Hash>
        + Hash
        + Ord
        + FromUniformBytes<64>,
{
    #[cfg(not(feature = "committed-instances"))]
    let nb_committed_instances: usize = 0;

    let domain = pk.get_vk().get_domain();
    let meta = pk.get_vk().cs();

    let ProverTrace {
        advice_polys,
        instance_polys,
        lookups,
        trashcans,
        permutations,
        challenges,
        y,
        beta,
        gamma,
        theta,
        trash_challenge,
        vanishing,
        ..
    } = trace;

    /*
    let g_coset_value = domain.g_coset;
    let g_coset_inv_value: F = g_coset_value.square(); 
    
    // Calculate the advice and instance cosets
    let advice_cosets: Vec<Vec<Polynomial<F, ExtendedLagrangeCoeff>>> = advice_polys
        .iter()
        .map(|advice_polys| {
            advice_polys
                .iter()
                .map(|poly| 
                {
                    let mut values_in = domain.empty_extended();
                    let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(poly.values.len()); 
                    let mut gpu_poly_extended = crate::DeviceMemPool::allocate::<F>(domain.extended_len()); 
                    crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &poly.values);
                    crate::gpu_coeff_to_extended(&mut gpu_poly_extended,&gpu_poly, &g_coset_value, &g_coset_inv_value);        
                    crate::DeviceMemPool::mem_copy_dtoh(&mut values_in.values, &gpu_poly_extended); 
                    crate::DeviceMemPool::deallocate(gpu_poly);
                    crate::DeviceMemPool::deallocate(gpu_poly_extended);
                    values_in
                }                  
                )
                .collect()
        })
        .collect();
    let instance_cosets: Vec<Vec<Polynomial<F, ExtendedLagrangeCoeff>>> = instance_polys
        .iter()
        .map(|instance_polys| {
            instance_polys
                .iter()
                .map(|poly|
                {
                    let mut values_in = domain.empty_extended();
                    let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(poly.values.len()); 
                    let mut gpu_poly_extended = crate::DeviceMemPool::allocate::<F>(domain.extended_len()); 
                    crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &poly.values);
                    crate::gpu_coeff_to_extended(&mut gpu_poly_extended,&gpu_poly, &g_coset_value, &g_coset_inv_value);        
                    crate::DeviceMemPool::mem_copy_dtoh(&mut values_in.values, &gpu_poly_extended); 
                    crate::DeviceMemPool::deallocate(gpu_poly);
                    crate::DeviceMemPool::deallocate(gpu_poly_extended);
                    values_in
                })
                .collect()
        })
        .collect();

    // Evaluate the h(X) polynomial
    let h_poly = pk.ev.evaluate_h::<ExtendedLagrangeCoeff>(
        &pk.vk.domain,
        &pk.vk.cs,
        &advice_cosets
            .iter()
            .map(|a| a.as_slice())
            .collect::<Vec<_>>(),
        &instance_cosets
            .iter()
            .map(|i| i.as_slice())
            .collect::<Vec<_>>(),
        &pk.fixed_cosets,
        &challenges,
        y,
        beta,
        gamma,
        theta,
        trash_challenge,
        &lookups,
        &trashcans,
        &permutations,
        &pk.l0,
        &pk.l_last,
        &pk.l_active_row,
        &pk.permutation.cosets,
    );

    // Advice and instance cosets are no longer required and are dropped to reduce
    // memory usage.
    drop(advice_cosets);
    drop(instance_cosets);
    */

    let g_coset_value = domain.g_coset;
    let g_coset_inv_value: F = g_coset_value.square(); 
    
    // Calculate the advice and instance cosets
    let gpu_advice_cosets_ptr: Vec<Vec<crate::GpuVec>> = advice_polys
        .iter()
        .map(|advice_polys| {
            advice_polys
                .iter()
                .map(|poly| 
                {
                    let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(poly.values.len()); 
                    let mut gpu_poly_extended = crate::DeviceMemPool::allocate::<F>(domain.extended_len()); 
                    crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &poly.values);
                    crate::gpu_coeff_to_extended(&mut gpu_poly_extended,&gpu_poly, &g_coset_value, &g_coset_inv_value);        
                    crate::DeviceMemPool::deallocate(gpu_poly);
                    gpu_poly_extended
                }                  
                )
                .collect()
        })
        .collect();
    let gpu_instance_cosets_ptr: Vec<Vec<crate::GpuVec>> = instance_polys
        .iter()
        .map(|instance_polys| {
            instance_polys
                .iter()
                .map(|poly|
                {
                    let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(poly.values.len()); 
                    let mut gpu_poly_extended = crate::DeviceMemPool::allocate::<F>(domain.extended_len()); 
                    crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &poly.values);
                    crate::gpu_coeff_to_extended(&mut gpu_poly_extended,&gpu_poly, &g_coset_value, &g_coset_inv_value);        
                    crate::DeviceMemPool::deallocate(gpu_poly);
                    gpu_poly_extended
                })
                .collect()
        })
        .collect();

    // Evaluate the h(X) polynomial
    let h_poly = pk.ev.evaluate_h_gpu::<ExtendedLagrangeCoeff>(
        &pk.vk.domain,
        &pk.vk.cs,
        &gpu_advice_cosets_ptr
            .iter()
            .map(|a| a.as_slice())
            .collect::<Vec<_>>(),
        &gpu_instance_cosets_ptr
            .iter()
            .map(|i| i.as_slice())
            .collect::<Vec<_>>(),
        &pk.fixed_cosets,
        &challenges,
        y,
        beta,
        gamma,
        theta,
        trash_challenge,
        &lookups,
        &trashcans,
        &permutations,
        &pk.l0,
        &pk.l_last,
        &pk.l_active_row,
        &pk.permutation.cosets,
    );

    // Advice and instance cosets are no longer required and are dropped to reduce
    // memory usage.
    for dev_vec in gpu_advice_cosets_ptr {
        for dev_ptr in dev_vec {
            crate::DeviceMemPool::deallocate(dev_ptr);
        }
    }

    for dev_vec in gpu_instance_cosets_ptr {
        for dev_ptr in dev_vec {
            crate::DeviceMemPool::deallocate(dev_ptr);
        }
    }

    // Construct the vanishing argument's h(X) commitments
    let vanishing =
        vanishing.construct::<CS, T>(params, pk.get_vk().get_domain(), h_poly, transcript)?;
    let start = std::time::Instant::now();

    let x: F = transcript.squeeze_challenge();

    // Compute and hash evals for the polynomials of the committed instances of
    // each circuit
    for instance in instance_polys.iter() {
        // Evaluate polynomials at omega^i x
        for &(column, at) in meta.instance_queries.iter() {
            if column.index() < nb_committed_instances {
                // let eval = eval_polynomial(&instance[column.index()], domain.rotate_omega(x, at));
                let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(instance[column.index()].len());
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &instance[column.index()]);
                let mut gpu_eval_res = crate::DeviceMemPool::allocate::<F>(1); 
                let x = domain.rotate_omega(x, at);
                crate::gpu_eval_polynomial(&gpu_poly, &x, &mut gpu_eval_res);
                let mut eval_values = [F::ZERO; 1];
                crate::DeviceMemPool::mem_copy_dtoh(&mut eval_values, &gpu_eval_res);
                let eval = eval_values[0];

                transcript.write(&eval)?;
            }
        }
    }

    // Compute and hash advice evals for each circuit instance
    for advice in advice_polys.iter() {
        // Evaluate polynomials at omega^i x
        let advice_evals: Vec<_> = meta
            .advice_queries
            .iter()
            .map(|&(column, at)| {
                // eval_polynomial(&advice[column.index()], domain.rotate_omega(x, at))
                let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(advice[column.index()].len());
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &advice[column.index()]);
                let mut gpu_eval_res = crate::DeviceMemPool::allocate::<F>(1); 
                let x = domain.rotate_omega(x, at);
                crate::gpu_eval_polynomial(&gpu_poly, &x, &mut gpu_eval_res);
                let mut eval_values = [F::ZERO; 1];
                crate::DeviceMemPool::mem_copy_dtoh(&mut eval_values, &gpu_eval_res);
                let eval = eval_values[0];
                eval

            })
            .collect();

        // Hash each advice column evaluation
        for eval in advice_evals.iter() {
            transcript.write(eval)?;
        }
    }

    // Compute and hash fixed evals (shared across all circuit instances)
    let fixed_evals: Vec<_> = meta
        .fixed_queries
        .iter()
        .map(|&(column, at)| {

            // eval_polynomial(&pk.fixed_polys[column.index()], domain.rotate_omega(x, at))
            let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(pk.fixed_polys[column.index()].len()); 
            crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &pk.fixed_polys[column.index()]); 
            let mut gpu_eval_res = crate::DeviceMemPool::allocate::<F>(1); 
            let x = domain.rotate_omega(x, at);
            crate::gpu_eval_polynomial(&gpu_poly, &x, &mut gpu_eval_res);
            let mut eval_values = [F::ZERO; 1];
            crate::DeviceMemPool::mem_copy_dtoh(&mut eval_values, &gpu_eval_res);
            let eval = eval_values[0];
            eval
        })
        .collect();

    // Hash each fixed column evaluation
    for eval in fixed_evals.iter() {
        transcript.write(eval)?;
    }

    let vanishing = vanishing.evaluate(x, domain, transcript)?;

    // Evaluate common permutation data
    pk.permutation.evaluate(x, transcript)?;

    // Evaluate the permutations, if any, at omega^i x.
    let permutations: Vec<permutation::prover::Evaluated<F>> = permutations
        .into_iter()
        .map(|permutation| -> Result<_, _> { permutation.evaluate(pk, x, transcript) })
        .collect::<Result<Vec<_>, _>>()?;

    // Evaluate the lookups, if any, at omega^i x.
    let lookups: Vec<Vec<lookup::prover::Evaluated<F>>> = lookups
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            lookups
                .into_iter()
                .map(|p| p.evaluate(pk, x, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Evaluate the trashcans, if any, at x.
    let trashcans: Vec<Vec<trash::prover::Evaluated<F>>> = trashcans
        .into_iter()
        .map(|trash| -> Result<Vec<_>, _> {
            trash
                .into_iter()
                .map(|p| p.evaluate(x, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let queries = instance_polys
        .iter()
        .zip(advice_polys.iter())
        .zip(permutations.iter())
        .zip(lookups.iter())
        .zip(trashcans.iter())
        .flat_map(|((((instance, advice), permutation), lookups), trash)| {
            iter::empty()
                .chain(
                    pk.vk
                        .cs
                        .instance_queries
                        .iter()
                        .take(nb_committed_instances)
                        .map(move |&(column, at)| ProverQuery {
                            point: domain.rotate_omega(x, at),
                            poly: &instance[column.index()],
                        }),
                )
                .chain(
                    pk.vk
                        .cs
                        .advice_queries
                        .iter()
                        .map(move |&(column, at)| ProverQuery {
                            point: domain.rotate_omega(x, at),
                            poly: &advice[column.index()],
                        }),
                )
                .chain(permutation.open(pk, x))
                .chain(lookups.iter().flat_map(move |p| p.open(pk, x)))
                .chain(trash.iter().flat_map(move |p| p.open(x)))
        })
        .chain(
            pk.vk
                .cs
                .fixed_queries
                .iter()
                .map(|&(column, at)| ProverQuery {
                    point: domain.rotate_omega(x, at),
                    poly: &pk.fixed_polys[column.index()],
                }),
        )
        .chain(pk.permutation.open(x))
        // We query the h(X) polynomial at x
        .chain(vanishing.open(x))
        .collect::<Vec<_>>();
    println!("Vanishing argument evaluation took: {:?}", start.elapsed());

    CS::multi_open(params, &queries, transcript).map_err(|_| Error::ConstraintSystemFailure)
}

/// This creates a proof for the provided `circuit` when given the public
/// parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
pub fn create_proof<
    F,
    CS: PolynomialCommitmentScheme<F>,
    T: Transcript,
    ConcreteCircuit: Circuit<F>,
>(
    params: &CS::Parameters,
    pk: &ProvingKey<F, CS>,
    circuits: &[ConcreteCircuit],
    #[cfg(feature = "committed-instances")] nb_committed_instances: usize,
    instances: &[&[&[F]]],
    rng: impl RngCore + CryptoRng,
    transcript: &mut T,
) -> Result<(), Error>
where
    CS::Commitment: Hashable<T::Hash>,
    F: WithSmallOrderMulGroup<3>
        + Sampleable<T::Hash>
        + Hashable<T::Hash>
        + Hash
        + Ord
        + FromUniformBytes<64>,
{   
    println!("GPU memory pool initializing with 0 bytes");
    crate::DeviceMemPool::initialize(0);
    println!("GPU memory pool initialized with 0 bytes");
    let start = std::time::Instant::now();
    let trace = compute_trace(
        params,
        pk,
        circuits,
        #[cfg(feature = "committed-instances")]
        nb_committed_instances,
        instances,
        rng,
        transcript,
    )?;
    println!("Trace computation took: {:?}", start.elapsed());
    finalise_proof(
        params,
        pk,
        #[cfg(feature = "committed-instances")]
        nb_committed_instances,
        trace,
        transcript,
    )
}

#[test]
fn test_create_proof() {
    use halo2curves::bn256::{Bn256, Fr};
    use rand_core::OsRng;

    use crate::{
        circuit::SimpleFloorPlanner,
        plonk::{keygen_pk, keygen_vk},
        poly::kzg::{params::ParamsKZG, KZGCommitmentScheme},
        transcript::CircuitTranscript,
    };

    #[derive(Clone, Copy)]
    struct MyCircuit;

    impl<F: Field> Circuit<F> for MyCircuit {
        type Config = ();
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            *self
        }

        fn configure(_meta: &mut ConstraintSystem<F>) -> Self::Config {}

        fn synthesize(
            &self,
            _config: Self::Config,
            _layouter: impl crate::circuit::Layouter<F>,
        ) -> Result<(), Error> {
            Ok(())
        }
    }

    let params: ParamsKZG<Bn256> = ParamsKZG::unsafe_setup(3, OsRng);
    let vk = keygen_vk(&params, &MyCircuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(vk, &MyCircuit).expect("keygen_pk should not fail");
    let mut transcript = CircuitTranscript::<_>::init();

    // Create proof with wrong number of instances
    let proof = create_proof::<Fr, KZGCommitmentScheme<Bn256>, _, _>(
        &params,
        &pk,
        &[MyCircuit, MyCircuit],
        #[cfg(feature = "committed-instances")]
        0,
        &[],
        OsRng,
        &mut transcript,
    );
    assert!(matches!(proof.unwrap_err(), Error::InvalidInstances));

    // Create proof with correct number of instances
    create_proof::<Fr, KZGCommitmentScheme<Bn256>, _, _>(
        &params,
        &pk,
        &[MyCircuit, MyCircuit],
        #[cfg(feature = "committed-instances")]
        0,
        &[&[], &[]],
        OsRng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
}
