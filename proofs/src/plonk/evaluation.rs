use ff::{PrimeField, WithSmallOrderMulGroup};
use group::ff::Field;

use super::{ConstraintSystem, Expression};
use crate::{
    plonk::{lookup, permutation, trash, Any},
    poly::{EvaluationDomain, Polynomial, PolynomialRepresentation, Rotation},
    utils::arithmetic::parallelize,
};

/// Return the index in the polynomial of size `isize` after rotation `rot`.
pub(crate) fn get_rotation_idx(idx: usize, rot: i32, rot_scale: i32, isize: i32) -> usize {
    (((idx as i32) + (rot * rot_scale)).rem_euclid(isize)) as usize
}

/// Value used in a calculation
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd)]
pub enum ValueSource {
    /// This is a constant value
    Constant(usize),
    /// This is an intermediate value
    Intermediate(usize),
    /// This is a fixed column
    Fixed(usize, usize),
    /// This is an advice (witness) column
    Advice(usize, usize),
    /// This is an instance (external) column
    Instance(usize, usize),
    /// This is a challenge
    Challenge(usize),
    /// beta
    Beta(),
    /// gamma
    Gamma(),
    /// theta
    Theta(),
    /// trash challenge
    TrashChallenge(),
    /// y
    Y(),
    /// Previous value
    PreviousValue(),
}

impl Default for ValueSource {
    fn default() -> Self {
        ValueSource::Constant(0)
    }
}

impl ValueSource {
    /// Get the value for this source
    #[allow(clippy::too_many_arguments)]
    pub fn get<F: Field, B: PolynomialRepresentation>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        trash_challenge: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        match self {
            ValueSource::Constant(idx) => constants[*idx],
            ValueSource::Intermediate(idx) => intermediates[*idx],
            ValueSource::Fixed(column_index, rotation) => {
                fixed_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Advice(column_index, rotation) => {
                advice_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Instance(column_index, rotation) => {
                instance_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Challenge(index) => challenges[*index],
            ValueSource::Beta() => *beta,
            ValueSource::Gamma() => *gamma,
            ValueSource::Theta() => *theta,
            ValueSource::TrashChallenge() => *trash_challenge,
            ValueSource::Y() => *y,
            ValueSource::PreviousValue() => *previous_value,
        }
    }

    pub fn to_ffi(&self) -> crate::ValueSourceFFI {
        match self {
            ValueSource::Constant(i) => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Constant,
                param0: *i,
                param1: 0,
            },
            ValueSource::Intermediate(i) => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Intermediate,
                param0: *i,
                param1: 0,
            },
            ValueSource::Fixed(col, rot) => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Fixed,
                param0: *col,
                param1: *rot,
            },
            ValueSource::Advice(col, rot) => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Advice,
                param0: *col,
                param1: *rot,
            },
            ValueSource::Instance(col, rot) => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Instance,
                param0: *col,
                param1: *rot,
            },
            ValueSource::Challenge(i) => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Challenge,
                param0: *i,
                param1: 0,
            },
            ValueSource::Beta() => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Beta,
                param0: 0,
                param1: 0,
            },
            ValueSource::Gamma() => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Gamma,
                param0: 0,
                param1: 0,
            },
            ValueSource::Theta() => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Theta,
                param0: 0,
                param1: 0,
            },
            ValueSource::TrashChallenge() => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::TrashChallenge,
                param0: 0,
                param1: 0,
            },
            ValueSource::Y() => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::Y,
                param0: 0,
                param1: 0,
            },
            ValueSource::PreviousValue() => crate::ValueSourceFFI {
                kind: crate::ValueSourceKind::PreviousValue,
                param0: 0,
                param1: 0,
            },
        }
    }
}

/// Calculation
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Calculation {
    /// This is an addition
    Add(ValueSource, ValueSource),
    /// This is a subtraction
    Sub(ValueSource, ValueSource),
    /// This is a product
    Mul(ValueSource, ValueSource),
    /// This is a square
    Square(ValueSource),
    /// This is a double
    Double(ValueSource),
    /// This is a negation
    Negate(ValueSource),
    /// This is Horner's rule: `val = a; val = val * c + b[]`
    Horner(ValueSource, Vec<ValueSource>, ValueSource),
    /// This is a simple assignment
    Store(ValueSource),
}

impl Calculation {
    /// Get the resulting value of this calculation
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<F: Field, B: PolynomialRepresentation>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        trash_challenge: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        let get_value = |value: &ValueSource| {
            value.get(
                rotations,
                constants,
                intermediates,
                fixed_values,
                advice_values,
                instance_values,
                challenges,
                beta,
                gamma,
                theta,
                trash_challenge,
                y,
                previous_value,
            )
        };
        match self {
            Calculation::Add(a, b) => get_value(a) + get_value(b),
            Calculation::Sub(a, b) => get_value(a) - get_value(b),
            Calculation::Mul(a, b) => get_value(a) * get_value(b),
            Calculation::Square(v) => get_value(v).square(),
            Calculation::Double(v) => get_value(v).double(),
            Calculation::Negate(v) => -get_value(v),
            Calculation::Horner(start_value, parts, factor) => {
                let factor = get_value(factor);
                let mut value = get_value(start_value);
                for part in parts.iter() {
                    value = value * factor + get_value(part);
                }
                value
            }
            Calculation::Store(v) => get_value(v),
        }
    }

    pub fn to_ffi(&self, arena: &mut Vec<crate::ValueSourceFFI>) -> crate::CalculationFFI {
        let dummy = crate::ValueSourceFFI {
            kind: crate::ValueSourceKind::Constant,
            param0: 0,
            param1: 0,
        };

        match self {
            Calculation::Add(a, b) => crate::CalculationFFI {
                kind: crate::CalculationKind::Add,
                a: a.to_ffi(),
                b: b.to_ffi(),
                extra: dummy,
                horner_parts_ptr: std::ptr::null(),
                horner_parts_len: 0,
            },
            Calculation::Sub(a, b) => crate::CalculationFFI {
                kind: crate::CalculationKind::Sub,
                a: a.to_ffi(),
                b: b.to_ffi(),
                extra: dummy,
                horner_parts_ptr: std::ptr::null(),
                horner_parts_len: 0,
            },
            Calculation::Mul(a, b) => crate::CalculationFFI {
                kind: crate::CalculationKind::Mul,
                a: a.to_ffi(),
                b: b.to_ffi(),
                extra: dummy,
                horner_parts_ptr: std::ptr::null(),
                horner_parts_len: 0,
            },
            Calculation::Square(a) => crate::CalculationFFI {
                kind: crate::CalculationKind::Square,
                a: a.to_ffi(),
                b: dummy,
                extra: dummy,
                horner_parts_ptr: std::ptr::null(),
                horner_parts_len: 0,
            },
            Calculation::Double(a) => crate::CalculationFFI {
                kind: crate::CalculationKind::Double,
                a: a.to_ffi(),
                b: dummy,
                extra: dummy,
                horner_parts_ptr: std::ptr::null(),
                horner_parts_len: 0,
            },
            Calculation::Negate(a) => crate::CalculationFFI {
                kind: crate::CalculationKind::Negate,
                a: a.to_ffi(),
                b: dummy,
                extra: dummy,
                horner_parts_ptr: std::ptr::null(),
                horner_parts_len: 0,
            },
            Calculation::Store(a) => crate::CalculationFFI {
                kind: crate::CalculationKind::Store,
                a: a.to_ffi(),
                b: dummy,
                extra: dummy,
                horner_parts_ptr: std::ptr::null(),
                horner_parts_len: 0,
            },
            Calculation::Horner(start, parts, factor) => {
                let start_ffi = start.to_ffi();
                let factor_ffi = factor.to_ffi();
                let offset = arena.len();
                arena.extend(parts.iter().map(|p| p.to_ffi()));
                let ptr = arena[offset..].as_ptr();
                let len = parts.len();
                crate::CalculationFFI {
                    kind: crate::CalculationKind::Horner,
                    a: start_ffi,
                    b: dummy,
                    extra: factor_ffi,
                    horner_parts_ptr: ptr,
                    horner_parts_len: len,
                }
            }
        }
    }
}

/// Evaluator
#[derive(Clone, Default, Debug)]
pub struct Evaluator<F: PrimeField> {
    ///  Custom gates evalution
    pub custom_gates: GraphEvaluator<F>,
    ///  Lookups evalution
    pub lookups: Vec<GraphEvaluator<F>>,
    ///  Trashcans evalution
    pub trashcans: Vec<GraphEvaluator<F>>,
}

/// GraphEvaluator
#[derive(Clone, Debug)]
pub struct GraphEvaluator<F: PrimeField> {
    /// Constants
    pub constants: Vec<F>,
    /// Rotations
    pub rotations: Vec<i32>,
    /// Calculations
    pub calculations: Vec<CalculationInfo>,
    /// Number of intermediates
    pub num_intermediates: usize,
}

/// EvaluationData
#[derive(Default, Debug)]
pub struct EvaluationData<F: PrimeField> {
    /// Intermediates
    pub intermediates: Vec<F>,
    /// Rotations
    pub rotations: Vec<usize>,
}

/// CaluclationInfo
#[derive(Clone, Debug)]
pub struct CalculationInfo {
    /// Calculation
    pub calculation: Calculation,
    /// Target
    pub target: usize,
}

impl CalculationInfo {
    pub fn to_ffi(&self, arena: &mut Vec<crate::ValueSourceFFI>) -> crate::CalculationInfoFFI {
        crate::CalculationInfoFFI {
            calculation: self.calculation.to_ffi(arena),
            target: self.target,
        }
    }
}

impl<F: WithSmallOrderMulGroup<3>> Evaluator<F> {
    /// Creates a new evaluation structure
    pub fn new(cs: &ConstraintSystem<F>) -> Self {
        let mut ev = Evaluator::default();

        // Custom gates
        let mut parts = Vec::new();
        for gate in cs.gates.iter() {
            parts.extend(
                gate.polynomials()
                    .iter()
                    .map(|poly| ev.custom_gates.add_expression(poly)),
            );
        }
        ev.custom_gates.add_calculation(Calculation::Horner(
            ValueSource::PreviousValue(),
            parts,
            ValueSource::Y(),
        ));

        // Lookups
        for lookup in cs.lookups.iter() {
            let mut graph = GraphEvaluator::default();

            let mut evaluate_lc = |expressions: &Vec<Expression<_>>| {
                let parts = expressions
                    .iter()
                    .map(|expr| graph.add_expression(expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            // Input coset
            let compressed_input_coset = evaluate_lc(&lookup.input_expressions);
            // table coset
            let compressed_table_coset = evaluate_lc(&lookup.table_expressions);
            // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            let right_gamma = graph.add_calculation(Calculation::Add(
                compressed_table_coset,
                ValueSource::Gamma(),
            ));
            let lc = graph.add_calculation(Calculation::Add(
                compressed_input_coset,
                ValueSource::Beta(),
            ));
            graph.add_calculation(Calculation::Mul(lc, right_gamma));

            ev.lookups.push(graph);
        }

        // Trashcans
        for trash in cs.trashcans.iter() {
            let mut graph = GraphEvaluator::default();

            let parts = trash
                .constraint_expressions()
                .iter()
                .map(|expr| graph.add_expression(expr))
                .collect();

            graph.add_calculation(Calculation::Horner(
                ValueSource::Constant(0),
                parts,
                ValueSource::TrashChallenge(),
            ));

            ev.trashcans.push(graph);
        }

        ev
    }

    /// Evaluate h poly
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_h<B: PolynomialRepresentation>(
        &self,
        domain: &EvaluationDomain<F>,
        cs: &ConstraintSystem<F>,
        advice: &[&[Polynomial<F, B>]],
        instance: &[&[Polynomial<F, B>]],
        fixed: &[Polynomial<F, B>],
        challenges: &[F],
        y: F,
        beta: F,
        gamma: F,
        theta: F,
        trash_challenge: F,
        lookups: &[Vec<lookup::prover::Committed<F>>],
        trashcans: &[Vec<trash::prover::Committed<F>>],
        permutations: &[permutation::prover::Committed<F>],
        l0: &Polynomial<F, B>,
        l_last: &Polynomial<F, B>,
        l_active_row: &Polynomial<F, B>,
        permutation_pk_cosets: &[Polynomial<F, B>],
    ) -> Polynomial<F, B> {
        let size = B::len(domain);
        let rot_scale = 1 << (B::k(domain) - domain.k());
        let omega = B::omega(domain);
        let isize = size as i32;
        let one = F::ONE;

        let p = &cs.permutation;

        let g_coset_value = domain.g_coset;
        let g_coset_inv_value: F = g_coset_value.square(); 

        let mut values = B::empty(domain);

        // Core expression evaluations
        let num_threads = rayon::current_num_threads();
        for ((((advice, instance), lookups), trashcans), permutation) in advice
            .iter()
            .zip(instance.iter())
            .zip(lookups.iter())
            .zip(trashcans.iter())
            .zip(permutations.iter())
        {
            // Custom gates
            /*
            rayon::scope(|scope| {
                let chunk_size = size.div_ceil(num_threads);
                for (thread_idx, values) in values.chunks_mut(chunk_size).enumerate() {
                    let start = thread_idx * chunk_size;
                    scope.spawn(move |_| {
                        let mut eval_data = self.custom_gates.instance();
                        for (i, value) in values.iter_mut().enumerate() {
                            let idx = start + i;
                            *value = self.custom_gates.evaluate::<B>(
                                &mut eval_data,
                                fixed,
                                advice,
                                instance,
                                challenges,
                                &beta,
                                &gamma,
                                &theta,
                                &trash_challenge,
                                &y,
                                value,
                                idx,
                                rot_scale,
                                isize,
                            );
                        }
                    });
                }
            });
            */

            let mut arena = Vec::new();
            let ffi_structs: Vec<crate::CalculationInfoFFI> = self.custom_gates.calculations
            .iter()
            .map(|c| c.to_ffi(&mut arena))
            .collect();

            let rotation_rot: Vec<i32> = self.custom_gates.rotations.iter().map(|rot| *rot).collect();

            let advice_gpu_vec: Vec<crate::GpuVec> = advice.iter().map(|poly|
            {
                let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(poly.values.len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &poly.values);
                gpu_poly
            } ).collect();

            let advice_poly_ptr: Vec<u64> = advice_gpu_vec.iter().map(|gpu_vec| gpu_vec.addr ).collect();

            let instance_gpu_vec: Vec<crate::GpuVec>= instance.iter().map(|poly|
            {
                let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(poly.values.len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &poly.values);
                gpu_poly
            } ).collect();

            let instance_poly_ptr: Vec<u64> = instance_gpu_vec.iter().map(|gpu_vec| gpu_vec.addr ).collect();

            let fixed_gpu_vec: Vec<crate::GpuVec>= fixed.iter().map(|poly|
            {
                let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(poly.values.len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &poly.values);
                gpu_poly
            } ).collect();

            let fixed_poly_ptr: Vec<u64> = fixed_gpu_vec.iter().map(|gpu_vec| gpu_vec.addr ).collect();

            let mut gpu_value = crate::DeviceMemPool::allocate::<F>(values.values.len()); 
            crate::DeviceMemPool::mem_copy_htod(&mut gpu_value, &values.values);

            crate::custom_gates_evaluation_r(&ffi_structs, &fixed_poly_ptr, &advice_poly_ptr, &instance_poly_ptr,
            challenges, &beta, &gamma, &theta, &trash_challenge, &y, &mut gpu_value, &self.custom_gates.constants, &rotation_rot, &rot_scale, &isize);

            //crate::DeviceMemPool::mem_copy_dtoh(&mut values.values, &gpu_value); 

            // Permutations
            let sets = &permutation.sets;
            if !sets.is_empty() {
                let blinding_factors = cs.blinding_factors();
                let last_rotation = Rotation(-((blinding_factors + 1) as i32));
                let chunk_len = cs.degree() - 2;
                let delta_start = beta * &B::g_coset(domain);

                let permutation_gpu_vec: Vec<crate::GpuVec> = sets
                    .iter()
                    //.map(|set| B::coeff_to_self(domain, set.permutation_product_poly.clone()))
                    .map(|set|{
                        let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(set.permutation_product_poly.values.len()); 
                        let mut gpu_poly_extended = crate::DeviceMemPool::allocate::<F>(domain.extended_len()); 
                        crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &set.permutation_product_poly.values);
                        crate::gpu_coeff_to_extended(&mut gpu_poly_extended,&gpu_poly, &g_coset_value, &g_coset_inv_value);        
                        crate::DeviceMemPool::deallocate(gpu_poly);
                        gpu_poly_extended
                    } )
                    .collect();

                let permutation_gpu_ptr: Vec<u64> = permutation_gpu_vec.iter().map(|gpu_vec| gpu_vec.addr ).collect();

                /*
                let first_set_permutation_product_coset =
                    permutation_product_cosets.first().unwrap();
                let last_set_permutation_product_coset = permutation_product_cosets.last().unwrap();

                // Permutation constraints
                parallelize(&mut values, |values, start| {
                    let mut beta_term = omega.pow_vartime([start as u64, 0, 0, 0]);
                    for (i, value) in values.iter_mut().enumerate() {
                        let idx = start + i;
                        let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                        let r_last = get_rotation_idx(idx, last_rotation.0, rot_scale, isize);

                        // Enforce only for the first set.
                        // l_0(X) * (1 - z_0(X)) = 0
                        *value = *value * y
                            + ((one - first_set_permutation_product_coset[idx]) * l0[idx]);
                        // Enforce only for the last set.
                        // l_last(X) * (z_l(X)^2 - z_l(X)) = 0
                        *value = *value * y
                            + ((last_set_permutation_product_coset[idx]
                                * last_set_permutation_product_coset[idx]
                                - last_set_permutation_product_coset[idx])
                                * l_last[idx]);
                        // Except for the first set, enforce.
                        // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
                        for set_idx in 0..sets.len() {
                            if set_idx != 0 {
                                *value = *value * y
                                    + ((permutation_product_cosets[set_idx][idx]
                                        - permutation_product_cosets[set_idx - 1][r_last])
                                        * l0[idx]);
                            }
                        }
                        // And for all the sets we enforce:
                        // (1 - (l_last(X) + l_blind(X))) * (
                        //   z_i(\omega X) \prod_j (p(X) + \beta s_j(X) + \gamma)
                        // - z_i(X) \prod_j (p(X) + \delta^j \beta X + \gamma)
                        // )
                        let mut current_delta = delta_start * beta_term;
                        for ((permutation_product_coset, columns), cosets) in
                            permutation_product_cosets
                                .iter()
                                .zip(p.columns.chunks(chunk_len))
                                .zip(permutation_pk_cosets.chunks(chunk_len))
                        {
                            let mut left = permutation_product_coset[r_next];
                            for (values, permutation) in columns
                                .iter()
                                .map(|&column| match column.column_type() {
                                    Any::Advice(_) => &advice[column.index()],
                                    Any::Fixed => &fixed[column.index()],
                                    Any::Instance => &instance[column.index()],
                                })
                                .zip(cosets.iter())
                            {
                                left *= values[idx] + beta * permutation[idx] + gamma;
                            }

                            let mut right = permutation_product_coset[idx];
                            for values in columns.iter().map(|&column| match column.column_type() {
                                Any::Advice(_) => &advice[column.index()],
                                Any::Fixed => &fixed[column.index()],
                                Any::Instance => &instance[column.index()],
                            }) {
                                right *= values[idx] + current_delta + gamma;
                                current_delta *= &F::DELTA;
                            }

                            *value = *value * y + ((left - right) * l_active_row[idx]);
                        }
                        beta_term *= &omega;
                    }
                });
                */

                let columns_ffi_structs: Vec<crate::ColumnFFI> = p.columns
                .iter()
                .map(|c| c.to_ffi())
                .collect();

                let mut gpu_l0 = crate::DeviceMemPool::allocate::<F>(l0.values.len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_l0, &l0.values);

                let mut gpu_l_last = crate::DeviceMemPool::allocate::<F>(l_last.values.len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_l_last, &l_last.values);

                let mut gpu_l_active_row = crate::DeviceMemPool::allocate::<F>(l_active_row.values.len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_l_active_row, &l_active_row.values);

                let permutation_coset_gpu_vec: Vec<crate::GpuVec>= permutation_pk_cosets.iter().map(|poly|
                {
                    let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(poly.values.len()); 
                    crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &poly.values);
                    gpu_poly
                } ).collect();

                let permutation_coset_gpu_ptr: Vec<u64> = permutation_coset_gpu_vec.iter().map(|gpu_vec| gpu_vec.addr ).collect();
                let chunk_len32: i32 = chunk_len as i32;

                crate::permutations_evaluation_r(&columns_ffi_structs, &fixed_poly_ptr, &advice_poly_ptr, &instance_poly_ptr,
                &mut gpu_value, & gpu_l0, &gpu_l_last, &gpu_l_active_row, &permutation_coset_gpu_ptr, &permutation_gpu_ptr,
                &delta_start, &F::DELTA, &beta, &gamma, &y, &omega, &chunk_len32, &last_rotation.0,
                &rot_scale, &isize);

                //crate::DeviceMemPool::mem_copy_dtoh(&mut values.values, &gpu_value); 
                
            }

            // Lookups
            for (n, lookup) in lookups.iter().enumerate() {
                // Polynomials required for this lookup.
                // Calculated here so these only have to be kept in memory for the short time
                // they are actually needed.

                //let product_coset = B::coeff_to_self(domain, lookup.product_poly.clone());
                //let permuted_input_coset =
                //    B::coeff_to_self(domain, lookup.permuted_input_poly.clone());
                //let permuted_table_coset =
                //    B::coeff_to_self(domain, lookup.permuted_table_poly.clone());

                ///////////////////////////
                 
                //let mut product_coset = B::empty(domain);
                let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(lookup.product_poly.values.len()); 
                let mut gpu_product_coset = crate::DeviceMemPool::allocate::<F>(domain.extended_len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &lookup.product_poly.values);
                crate::gpu_coeff_to_extended(&mut gpu_product_coset,&gpu_poly, &g_coset_value, &g_coset_inv_value);        
                //crate::DeviceMemPool::mem_copy_dtoh(&mut product_coset.values, &gpu_product_coset); 
                crate::DeviceMemPool::deallocate(gpu_poly);
                //crate::DeviceMemPool::deallocate(gpu_product_coset);

                //

                //let mut permuted_input_coset = B::empty(domain);
                let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(lookup.permuted_input_poly.values.len()); 
                let mut gpu_permuted_input_coset = crate::DeviceMemPool::allocate::<F>(domain.extended_len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &lookup.permuted_input_poly.values);
                crate::gpu_coeff_to_extended(&mut gpu_permuted_input_coset,&gpu_poly, &g_coset_value, &g_coset_inv_value);        
                //crate::DeviceMemPool::mem_copy_dtoh(&mut permuted_input_coset.values, &gpu_permuted_input_coset); 
                crate::DeviceMemPool::deallocate(gpu_poly);
                //crate::DeviceMemPool::deallocate(gpu_permuted_input_coset);

                //

                //let mut permuted_table_coset = B::empty(domain);
                let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(lookup.permuted_table_poly.values.len()); 
                let mut gpu_permuted_table_coset = crate::DeviceMemPool::allocate::<F>(domain.extended_len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &lookup.permuted_table_poly.values);
                crate::gpu_coeff_to_extended(&mut gpu_permuted_table_coset,&gpu_poly, &g_coset_value, &g_coset_inv_value);        
                //crate::DeviceMemPool::mem_copy_dtoh(&mut permuted_table_coset.values, &gpu_permuted_table_coset); 
                crate::DeviceMemPool::deallocate(gpu_poly);
                //crate::DeviceMemPool::deallocate(gpu_permuted_table_coset);
 
                ///////////////////////////
                
                let mut arena_lookups = Vec::with_capacity(1024);
                let ffi_structs_lookups: Vec<crate::CalculationInfoFFI> = self.lookups[n].calculations
                .iter()
                .map(|c| c.to_ffi(&mut arena_lookups))
                .collect();

                let rotation_rot_lookups: Vec<i32> = self.lookups[n].rotations.iter().map(|rot| *rot).collect();

                let mut gpu_l0 = crate::DeviceMemPool::allocate::<F>(l0.values.len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_l0, &l0.values);

                let mut gpu_l_last = crate::DeviceMemPool::allocate::<F>(l_last.values.len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_l_last, &l_last.values);

                let mut gpu_l_active_row = crate::DeviceMemPool::allocate::<F>(l_active_row.values.len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_l_active_row, &l_active_row.values);

                crate::lookups_evaluation_r(&ffi_structs_lookups, &fixed_poly_ptr, &advice_poly_ptr, &instance_poly_ptr,
                &gpu_l0, &gpu_l_last, &gpu_l_active_row, &mut gpu_value, 
                &gpu_product_coset, &gpu_permuted_input_coset ,& gpu_permuted_table_coset,
                &challenges, &beta,
                &gamma, &theta, &trash_challenge, &y,&self.lookups[n].constants, &rotation_rot_lookups,
                 &rot_scale, &isize);
            
                /*
                // Lookup constraints
                parallelize(&mut values, |values, start| {
                    let lookup_evaluator = &self.lookups[n];
                    let mut eval_data = lookup_evaluator.instance();
                    for (i, value) in values.iter_mut().enumerate() {
                        let idx = start + i;

                        let table_value = lookup_evaluator.evaluate(
                            &mut eval_data,
                            fixed,
                            advice,
                            instance,
                            challenges,
                            &beta,
                            &gamma,
                            &theta,
                            &trash_challenge,
                            &y,
                            &F::ZERO,
                            idx,
                            rot_scale,
                            isize,
                        );

                        let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                        let r_prev = get_rotation_idx(idx, -1, rot_scale, isize);

                        let a_minus_s = permuted_input_coset[idx] - permuted_table_coset[idx];
                        // l_0(X) * (1 - z(X)) = 0
                        *value = *value * y + ((one - product_coset[idx]) * l0[idx]);
                        // l_last(X) * (z(X)^2 - z(X)) = 0
                        *value = *value * y
                            + ((product_coset[idx] * product_coset[idx] - product_coset[idx])
                                * l_last[idx]);
                        // (1 - (l_last(X) + l_blind(X))) * (
                        //   z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
                        //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta) (\theta^{m-1}
                        //     s_0(X) + ... + s_{m-1}(X) + \gamma)
                        // ) = 0
                        *value = *value * y
                            + ((product_coset[r_next]
                                * (permuted_input_coset[idx] + beta)
                                * (permuted_table_coset[idx] + gamma)
                                - product_coset[idx] * table_value)
                                * l_active_row[idx]);
                        // Check that the first values in the permuted input expression and permuted
                        // fixed expression are the same.
                        // l_0(X) * (a'(X) - s'(X)) = 0
                        *value = *value * y + (a_minus_s * l0[idx]);
                        // Check that each value in the permuted lookup input expression is either
                        // equal to the value above it, or the value at the same index in the
                        // permuted table expression.
                        // (1 - (l_last + l_blind)) * (a′(X) − s′(X))⋅(a′(X) − a′(\omega^{-1} X)) =
                        // 0
                        *value = *value * y
                            + (a_minus_s
                                * (permuted_input_coset[idx] - permuted_input_coset[r_prev])
                                * l_active_row[idx]);
                    }
                });
                */
            }

            crate::DeviceMemPool::mem_copy_dtoh(&mut values.values, &gpu_value); 

            // Trashcans
            for (n, trash) in trashcans.iter().enumerate() {
                // Polynomials required for this trash argument.
                // Calculated here so these only have to be kept in memory for the short time
                // they are actually needed.
                let trash_poly = B::coeff_to_self(domain, trash.trash_poly.clone());

                // Trash argument constraints.
                parallelize(&mut values, |values, start| {
                    let trash_evaluator = &self.trashcans[n];
                    let argument = &cs.trashcans[n];
                    let mut eval_data = trash_evaluator.instance();
                    for (i, value) in values.iter_mut().enumerate() {
                        let idx = start + i;

                        let compressed_expression = trash_evaluator.evaluate(
                            &mut eval_data,
                            fixed,
                            advice,
                            instance,
                            challenges,
                            &beta,
                            &gamma,
                            &theta,
                            &trash_challenge,
                            &y,
                            &F::ZERO,
                            idx,
                            rot_scale,
                            isize,
                        );

                        let q = match argument.selector() {
                            Expression::Fixed(query) => fixed[query.index().unwrap()][idx],
                            _ => unreachable!(),
                        };

                        // compressed_expressions - (1 - q) * trash
                        *value = *value * y + (compressed_expression - (one - q) * trash_poly[idx]);
                    }
                });
            }
        }
        values
    }

    /// Evaluate h poly (GPU version)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_h_gpu<B: PolynomialRepresentation>(
        &self,
        domain: &EvaluationDomain<F>,
        cs: &ConstraintSystem<F>,
        advice: &[&[crate::GpuVec]],
        instance: &[&[crate::GpuVec]],
        fixed: &[Polynomial<F, B>],
        challenges: &[F],
        y: F,
        beta: F,
        gamma: F,
        theta: F,
        trash_challenge: F,
        lookups: &[Vec<lookup::prover::Committed<F>>],
        trashcans: &[Vec<trash::prover::Committed<F>>],
        permutations: &[permutation::prover::Committed<F>],
        l0: &Polynomial<F, B>,
        l_last: &Polynomial<F, B>,
        l_active_row: &Polynomial<F, B>,
        permutation_pk_cosets: &[Polynomial<F, B>],
    ) -> Polynomial<F, B> {
        let size = B::len(domain);
        let rot_scale = 1 << (B::k(domain) - domain.k());
        let omega = B::omega(domain);
        let isize = size as i32;
        let one = F::ONE;

        let p = &cs.permutation;

        let g_coset_value = domain.g_coset;
        let g_coset_inv_value: F = g_coset_value.square(); 

        let mut values = B::empty(domain);

        // Core expression evaluations
        let num_threads = rayon::current_num_threads();
        for ((((advice, instance), lookups), trashcans), permutation) in advice
            .iter()
            .zip(instance.iter())
            .zip(lookups.iter())
            .zip(trashcans.iter())
            .zip(permutations.iter())
        {
            // Custom gates
            let mut arena = Vec::new();
            let ffi_structs: Vec<crate::CalculationInfoFFI> = self.custom_gates.calculations
            .iter()
            .map(|c| c.to_ffi(&mut arena))
            .collect();

            let rotation_rot: Vec<i32> = self.custom_gates.rotations.iter().map(|rot| *rot).collect();

            //let advice_gpu_vec: Vec<crate::GpuVec> = advice.iter().map(|poly|
            //{
            //    let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(poly.values.len()); 
            //    crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &poly.values);
            //    gpu_poly
            //} ).collect();

            let advice_poly_ptr: Vec<u64> = advice.iter().map(|gpu_vec| gpu_vec.addr ).collect();

            //let instance_gpu_vec: Vec<crate::GpuVec>= instance.iter().map(|poly|
            //{
            //    let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(poly.values.len()); 
            //    crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &poly.values);
            //    gpu_poly
            //} ).collect();

            let instance_poly_ptr: Vec<u64> = instance.iter().map(|gpu_vec| gpu_vec.addr ).collect();

            let fixed_gpu_vec: Vec<crate::GpuVec>= fixed.iter().map(|poly|
            {
                let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(poly.values.len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &poly.values);
                gpu_poly
            } ).collect();

            let fixed_poly_ptr: Vec<u64> = fixed_gpu_vec.iter().map(|gpu_vec| gpu_vec.addr ).collect();

            let mut gpu_value = crate::DeviceMemPool::allocate::<F>(values.values.len()); 

            crate::custom_gates_evaluation_r(&ffi_structs, &fixed_poly_ptr, &advice_poly_ptr, &instance_poly_ptr,
            challenges, &beta, &gamma, &theta, &trash_challenge, &y, &mut gpu_value, &self.custom_gates.constants, &rotation_rot, &rot_scale, &isize);

            let mut gpu_l0 = crate::DeviceMemPool::allocate::<F>(l0.values.len()); 
            crate::DeviceMemPool::mem_copy_htod(&mut gpu_l0, &l0.values);

            let mut gpu_l_last = crate::DeviceMemPool::allocate::<F>(l_last.values.len()); 
            crate::DeviceMemPool::mem_copy_htod(&mut gpu_l_last, &l_last.values);

            let mut gpu_l_active_row = crate::DeviceMemPool::allocate::<F>(l_active_row.values.len()); 
            crate::DeviceMemPool::mem_copy_htod(&mut gpu_l_active_row, &l_active_row.values);

            // Permutations
            let sets = &permutation.sets;
            if !sets.is_empty() {
                let blinding_factors = cs.blinding_factors();
                let last_rotation = Rotation(-((blinding_factors + 1) as i32));
                let chunk_len = cs.degree() - 2;
                let delta_start = beta * &B::g_coset(domain);

                let permutation_gpu_vec: Vec<crate::GpuVec> = sets
                    .iter()
                    .map(|set|{
                        let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(set.permutation_product_poly.values.len()); 
                        let mut gpu_poly_extended = crate::DeviceMemPool::allocate::<F>(domain.extended_len()); 
                        crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &set.permutation_product_poly.values);
                        crate::gpu_coeff_to_extended(&mut gpu_poly_extended,&gpu_poly, &g_coset_value, &g_coset_inv_value);        
                        crate::DeviceMemPool::deallocate(gpu_poly);
                        gpu_poly_extended
                    } )
                    .collect();

                let permutation_gpu_ptr: Vec<u64> = permutation_gpu_vec.iter().map(|gpu_vec| gpu_vec.addr ).collect();

                let columns_ffi_structs: Vec<crate::ColumnFFI> = p.columns
                .iter()
                .map(|c| c.to_ffi())
                .collect();

                let permutation_coset_gpu_vec: Vec<crate::GpuVec>= permutation_pk_cosets.iter().map(|poly|
                {
                    let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(poly.values.len()); 
                    crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &poly.values);
                    gpu_poly
                } ).collect();

                let permutation_coset_gpu_ptr: Vec<u64> = permutation_coset_gpu_vec.iter().map(|gpu_vec| gpu_vec.addr ).collect();
                let chunk_len32: i32 = chunk_len as i32;

                crate::permutations_evaluation_r(&columns_ffi_structs, &fixed_poly_ptr, &advice_poly_ptr, &instance_poly_ptr,
                &mut gpu_value, & gpu_l0, &gpu_l_last, &gpu_l_active_row, &permutation_coset_gpu_ptr, &permutation_gpu_ptr,
                &delta_start, &F::DELTA, &beta, &gamma, &y, &omega, &chunk_len32, &last_rotation.0,
                &rot_scale, &isize);  

                for dev_vec in permutation_gpu_vec {
                    crate::DeviceMemPool::deallocate(dev_vec);
                }   

                for dev_vec in permutation_coset_gpu_vec {
                    crate::DeviceMemPool::deallocate(dev_vec);
                }    
            }

            // Lookups
            for (n, lookup) in lookups.iter().enumerate() {
                // Polynomials required for this lookup.
                // Calculated here so these only have to be kept in memory for the short time
                // they are actually needed.
                let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(lookup.product_poly.values.len()); 
                let mut gpu_product_coset = crate::DeviceMemPool::allocate::<F>(domain.extended_len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &lookup.product_poly.values);
                crate::gpu_coeff_to_extended(&mut gpu_product_coset,&gpu_poly, &g_coset_value, &g_coset_inv_value);        
                crate::DeviceMemPool::deallocate(gpu_poly);

                let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(lookup.permuted_input_poly.values.len()); 
                let mut gpu_permuted_input_coset = crate::DeviceMemPool::allocate::<F>(domain.extended_len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &lookup.permuted_input_poly.values);
                crate::gpu_coeff_to_extended(&mut gpu_permuted_input_coset,&gpu_poly, &g_coset_value, &g_coset_inv_value);        
                crate::DeviceMemPool::deallocate(gpu_poly);

                let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(lookup.permuted_table_poly.values.len()); 
                let mut gpu_permuted_table_coset = crate::DeviceMemPool::allocate::<F>(domain.extended_len()); 
                crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &lookup.permuted_table_poly.values);
                crate::gpu_coeff_to_extended(&mut gpu_permuted_table_coset,&gpu_poly, &g_coset_value, &g_coset_inv_value);        
                crate::DeviceMemPool::deallocate(gpu_poly);
                 
                let mut arena_lookups = Vec::with_capacity(1024);
                let ffi_structs_lookups: Vec<crate::CalculationInfoFFI> = self.lookups[n].calculations
                .iter()
                .map(|c| c.to_ffi(&mut arena_lookups))
                .collect();

                let rotation_rot_lookups: Vec<i32> = self.lookups[n].rotations.iter().map(|rot| *rot).collect();

                crate::lookups_evaluation_r(&ffi_structs_lookups, &fixed_poly_ptr, &advice_poly_ptr, &instance_poly_ptr,
                &gpu_l0, &gpu_l_last, &gpu_l_active_row, &mut gpu_value, 
                &gpu_product_coset, &gpu_permuted_input_coset ,& gpu_permuted_table_coset,
                &challenges, &beta,
                &gamma, &theta, &trash_challenge, &y,&self.lookups[n].constants, &rotation_rot_lookups,
                &rot_scale, &isize);

                crate::DeviceMemPool::deallocate(gpu_product_coset);
                crate::DeviceMemPool::deallocate(gpu_permuted_input_coset);
                crate::DeviceMemPool::deallocate(gpu_permuted_table_coset);
            }

            crate::DeviceMemPool::mem_copy_dtoh(&mut values.values, &gpu_value); 

            crate::DeviceMemPool::deallocate(gpu_value);
            crate::DeviceMemPool::deallocate(gpu_l0);
            crate::DeviceMemPool::deallocate(gpu_l_last);
            crate::DeviceMemPool::deallocate(gpu_l_active_row);
            for dev_vec in fixed_gpu_vec {
                crate::DeviceMemPool::deallocate(dev_vec);
            }    

            // Implement it on GPU!
            // Trashcans
            /*
            for (n, trash) in trashcans.iter().enumerate() {
                // Polynomials required for this trash argument.
                // Calculated here so these only have to be kept in memory for the short time
                // they are actually needed.
                let trash_poly = B::coeff_to_self(domain, trash.trash_poly.clone());

                // Trash argument constraints.
                parallelize(&mut values, |values, start| {
                    let trash_evaluator = &self.trashcans[n];
                    let argument = &cs.trashcans[n];
                    let mut eval_data = trash_evaluator.instance();
                    for (i, value) in values.iter_mut().enumerate() {
                        let idx = start + i;

                        let compressed_expression = trash_evaluator.evaluate(
                            &mut eval_data,
                            fixed,
                            advice,
                            instance,
                            challenges,
                            &beta,
                            &gamma,
                            &theta,
                            &trash_challenge,
                            &y,
                            &F::ZERO,
                            idx,
                            rot_scale,
                            isize,
                        );

                        let q = match argument.selector() {
                            Expression::Fixed(query) => fixed[query.index().unwrap()][idx],
                            _ => unreachable!(),
                        };

                        // compressed_expressions - (1 - q) * trash
                        *value = *value * y + (compressed_expression - (one - q) * trash_poly[idx]);
                    }
                });
            }
            */
        }
        values
    }

}

impl<F: PrimeField> Default for GraphEvaluator<F> {
    fn default() -> Self {
        Self {
            // Fixed positions to allow easy access
            constants: vec![F::ZERO, F::ONE, F::from(2u64)],
            rotations: Vec::new(),
            calculations: Vec::new(),
            num_intermediates: 0,
        }
    }
}

impl<F: PrimeField> GraphEvaluator<F> {
    /// Adds a rotation
    fn add_rotation(&mut self, rotation: &Rotation) -> usize {
        let position = self.rotations.iter().position(|&c| c == rotation.0);
        match position {
            Some(pos) => pos,
            None => {
                self.rotations.push(rotation.0);
                self.rotations.len() - 1
            }
        }
    }

    /// Adds a constant
    fn add_constant(&mut self, constant: &F) -> ValueSource {
        let position = self.constants.iter().position(|&c| c == *constant);
        ValueSource::Constant(match position {
            Some(pos) => pos,
            None => {
                self.constants.push(*constant);
                self.constants.len() - 1
            }
        })
    }

    /// Adds a calculation.
    /// Currently does the simplest thing possible: just stores the
    /// resulting value so the result can be reused  when that calculation
    /// is done multiple times.
    fn add_calculation(&mut self, calculation: Calculation) -> ValueSource {
        let existing_calculation = self
            .calculations
            .iter()
            .find(|c| c.calculation == calculation);
        match existing_calculation {
            Some(existing_calculation) => ValueSource::Intermediate(existing_calculation.target),
            None => {
                let target = self.num_intermediates;
                self.calculations.push(CalculationInfo {
                    calculation,
                    target,
                });
                self.num_intermediates += 1;
                ValueSource::Intermediate(target)
            }
        }
    }

    /// Generates an optimized evaluation for the expression
    fn add_expression(&mut self, expr: &Expression<F>) -> ValueSource {
        match expr {
            Expression::Constant(scalar) => self.add_constant(scalar),
            Expression::Selector(_selector) => unreachable!(),
            Expression::Fixed(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Fixed(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Advice(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Advice(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Instance(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Instance(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Challenge(challenge) => self.add_calculation(Calculation::Store(
                ValueSource::Challenge(challenge.index()),
            )),
            Expression::Negated(a) => match **a {
                Expression::Constant(scalar) => self.add_constant(&-scalar),
                _ => {
                    let result_a = self.add_expression(a);
                    match result_a {
                        ValueSource::Constant(0) => result_a,
                        _ => self.add_calculation(Calculation::Negate(result_a)),
                    }
                }
            },
            Expression::Sum(a, b) => {
                // Undo subtraction stored as a + (-b) in expressions
                match &**b {
                    Expression::Negated(b_int) => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b_int);
                        if result_a == ValueSource::Constant(0) {
                            self.add_calculation(Calculation::Negate(result_b))
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else {
                            self.add_calculation(Calculation::Sub(result_a, result_b))
                        }
                    }
                    _ => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b);
                        if result_a == ValueSource::Constant(0) {
                            result_b
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else if result_a <= result_b {
                            self.add_calculation(Calculation::Add(result_a, result_b))
                        } else {
                            self.add_calculation(Calculation::Add(result_b, result_a))
                        }
                    }
                }
            }
            Expression::Product(a, b) => {
                let result_a = self.add_expression(a);
                let result_b = self.add_expression(b);
                if result_a == ValueSource::Constant(0) || result_b == ValueSource::Constant(0) {
                    ValueSource::Constant(0)
                } else if result_a == ValueSource::Constant(1) {
                    result_b
                } else if result_b == ValueSource::Constant(1) {
                    result_a
                } else if result_a == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_b))
                } else if result_b == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_a))
                } else if result_a == result_b {
                    self.add_calculation(Calculation::Square(result_a))
                } else if result_a <= result_b {
                    self.add_calculation(Calculation::Mul(result_a, result_b))
                } else {
                    self.add_calculation(Calculation::Mul(result_b, result_a))
                }
            }
            Expression::Scaled(a, f) => {
                if *f == F::ZERO {
                    ValueSource::Constant(0)
                } else if *f == F::ONE {
                    self.add_expression(a)
                } else {
                    let cst = self.add_constant(f);
                    let result_a = self.add_expression(a);
                    self.add_calculation(Calculation::Mul(result_a, cst))
                }
            }
        }
    }

    /// Creates a new evaluation structure
    pub fn instance(&self) -> EvaluationData<F> {
        EvaluationData {
            intermediates: vec![F::ZERO; self.num_intermediates],
            rotations: vec![0usize; self.rotations.len()],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<B: PolynomialRepresentation>(
        &self,
        data: &mut EvaluationData<F>,
        fixed: &[Polynomial<F, B>],
        advice: &[Polynomial<F, B>],
        instance: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        trash_challenge: &F,
        y: &F,
        previous_value: &F,
        idx: usize,
        rot_scale: i32,
        isize: i32,
    ) -> F {
        // All rotation index values
        for (rot_idx, rot) in self.rotations.iter().enumerate() {
            data.rotations[rot_idx] = get_rotation_idx(idx, *rot, rot_scale, isize);
        }

        // All calculations, with cached intermediate results
        for calc in self.calculations.iter() {
            data.intermediates[calc.target] = calc.calculation.evaluate(
                &data.rotations,
                &self.constants,
                &data.intermediates,
                fixed,
                advice,
                instance,
                challenges,
                beta,
                gamma,
                theta,
                trash_challenge,
                y,
                previous_value,
            );
        }

        // Return the result of the last calculation (if any)
        if let Some(calc) = self.calculations.last() {
            data.intermediates[calc.target]
        } else {
            F::ZERO
        }
    }
}

/// Simple evaluation of an expression
pub fn evaluate<F: Field, B: PolynomialRepresentation>(
    expression: &Expression<F>,
    size: usize,
    rot_scale: i32,
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    challenges: &[F],
) -> Vec<F> {
    let mut values = vec![F::ZERO; size];
    let isize = size as i32;
    parallelize(&mut values, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = start + i;
            *value = expression.evaluate(
                &|scalar| scalar,
                &|_| panic!("virtual selectors are removed during optimization"),
                &|query| {
                    fixed[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    advice[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    instance[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|challenge| challenges[challenge.index()],
                &|a| -a,
                &|a, b| a + &b,
                &|a, b| a * b,
                &|a, scalar| a * scalar,
            );
        }
    });
    values
}
