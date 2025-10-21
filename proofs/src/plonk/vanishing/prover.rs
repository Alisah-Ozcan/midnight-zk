use std::{collections::HashMap, iter};
use std::marker::PhantomData;

use ff::{PrimeField, WithSmallOrderMulGroup};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use rayon::current_num_threads;

use super::Argument;
use crate::{
    plonk::Error,
    poly::{
        commitment::PolynomialCommitmentScheme, Coeff, EvaluationDomain, ExtendedLagrangeCoeff,
        Polynomial, ProverQuery,
    },
    transcript::{Hashable, Transcript},
    utils::arithmetic::{eval_polynomial, parallelize},
};

#[derive(Debug)]
pub(crate) struct Committed<F: PrimeField> {
    pub(crate) random_poly: Polynomial<F, Coeff>,
    random_poly_gpu: crate::GpuVec,
}

pub(crate) struct Constructed<F: PrimeField> {
    h_pieces: Vec<Polynomial<F, Coeff>>,
    committed: Committed<F>,
    h_pieces_gpu_flat: crate::GpuVec,
}

pub(crate) struct Evaluated<F: PrimeField> {
    h_poly: Polynomial<F, Coeff>,
    committed: Committed<F>,
}

impl<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>> Argument<F, CS> {
    pub(crate) fn commit<R: RngCore, T: Transcript>(
        params: &CS::Parameters,
        domain: &EvaluationDomain<F>,
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Committed<F>, Error>
    where
        CS::Commitment: Hashable<T::Hash>,
        F: Hashable<T::Hash>,
    {
        // Sample a random polynomial of degree n - 1
        let n = 1usize << domain.k() as usize;
        let mut rand_vec = vec![F::ZERO; n];

        let num_threads = current_num_threads();
        let chunk_size = n / num_threads;
        let thread_seeds = (0..)
            .step_by(chunk_size + 1)
            .take(n % num_threads)
            .chain(
                (chunk_size != 0)
                    .then(|| ((n % num_threads) * (chunk_size + 1)..).step_by(chunk_size))
                    .into_iter()
                    .flatten(),
            )
            .take(num_threads)
            .zip(iter::repeat_with(|| {
                let mut seed = [0u8; 32];
                rng.fill_bytes(&mut seed);
                ChaCha20Rng::from_seed(seed)
            }))
            .collect::<HashMap<_, _>>();

        parallelize(&mut rand_vec, |chunk, offset| {
            let mut rng = thread_seeds[&offset].clone();
            chunk.iter_mut().for_each(|v| *v = F::random(&mut rng));
        });

        let random_poly: Polynomial<F, Coeff> = domain.coeff_from_vec(rand_vec);

        // Commit
        //let c = CS::commit(params, &random_poly);
        let mut poly_gpu = crate::DeviceMemPool::allocate::<F>(random_poly.len()); 
        crate::DeviceMemPool::mem_copy_htod(&mut poly_gpu, &random_poly.values);     

        let mut poly_gpu_cp = crate::DeviceMemPool::allocate::<F>(random_poly.len()); 
        crate::DeviceMemPool::mem_copy_dtod(&mut poly_gpu_cp, &poly_gpu);

        let c = CS::commit_gpu(params, &poly_gpu);
        crate::DeviceMemPool::deallocate(poly_gpu);
                
        transcript.write(&c)?;

        Ok(Committed { random_poly, random_poly_gpu: poly_gpu_cp })
    }
}

impl<F: WithSmallOrderMulGroup<3>> Committed<F> {
    pub(crate) fn construct<CS: PolynomialCommitmentScheme<F>, T: Transcript>(
        self,
        params: &CS::Parameters,
        domain: &EvaluationDomain<F>,
        h_poly: Polynomial<F, ExtendedLagrangeCoeff>,
        transcript: &mut T,
    ) -> Result<Constructed<F>, Error>
    where
        CS::Commitment: Hashable<T::Hash>,
        F: Hashable<T::Hash>,
    {
        // Divide by t(X) = X^{params.n} - 1.
        let mut gpu_poly = crate::DeviceMemPool::allocate::<F>(domain.extended_len()); 
        crate::DeviceMemPool::mem_copy_htod(&mut gpu_poly, &h_poly.values);
        domain.divide_by_vanishing_poly_gpu(&gpu_poly);


        domain.extended_to_coeff_gpu(&mut gpu_poly);

        let mut h_poly = Vec::<F>::with_capacity(gpu_poly.len());
        #[allow(unsafe_code)]
        unsafe {
            h_poly.set_len(gpu_poly.len());
        }        
        crate::DeviceMemPool::mem_copy_dtoh(&mut h_poly, &gpu_poly); 


        let pieces_count = gpu_poly.len() / (domain.n as usize);
        let piece_size = domain.n as usize;

        let h_pieces_gpu: Vec<crate::GpuVec> = (0..pieces_count)
            .map(|i| {
                // Calculate offset for each piece in GPU memory
                let offset_bytes = i * piece_size * gpu_poly.elem_size;
                let piece_gpu_ptr = gpu_poly.addr + offset_bytes as u64;

                crate::GpuVec {
                    addr: piece_gpu_ptr,
                    size_bytes: piece_size * gpu_poly.elem_size,
                    elem_size: gpu_poly.elem_size,

                }
            })
            .collect();

        let mut h_pieces_gpu_flat = crate::DeviceMemPool::allocate::<F>(domain.extended_len());
        crate::DeviceMemPool::mem_copy_dtod(&mut h_pieces_gpu_flat, &gpu_poly);

        let h_commitments: Vec<_> = h_pieces_gpu
            .iter()
            .map(|h_piece| {
                CS::commit_gpu(params, &h_piece)
            })
            .collect();

        // Split h(X) up into pieces
        let h_pieces = h_poly
            .chunks_exact(domain.n as usize)
            .map(|v| domain.coeff_from_vec(v.to_vec()))
            .collect::<Vec<_>>();

        drop(h_poly);
        crate::DeviceMemPool::deallocate(gpu_poly);


        // Hash each h(X) piece
        for c in h_commitments {
            transcript.write(&c)?;
        }

        Ok(Constructed {
            h_pieces,
            committed: self,
            h_pieces_gpu_flat,
        })
    }
}

impl<F: WithSmallOrderMulGroup<3>> Constructed<F> {
    pub(crate) fn evaluate<T: Transcript>(
        self,
        x: F,
        domain: &EvaluationDomain<F>,
        transcript: &mut T,
    ) -> Result<Evaluated<F>, Error>
    where
        F: Hashable<T::Hash>,
    {
        let xn: F = x.pow_vartime([domain.n]);

        let h_piece_num = self.h_pieces.len();
        let poly_size = self.h_pieces[0].len();
        let mut gpu_h_pieces = self.h_pieces_gpu_flat;

        let mut gpu_h_poly = crate::DeviceMemPool::allocate::<F>(poly_size);
        crate::gpu_inner_product(&gpu_h_pieces, &xn, &mut gpu_h_poly, h_piece_num);
        let mut poly_values = Vec::<F>::with_capacity(poly_size);
        #[allow(unsafe_code)]
        unsafe {
            poly_values.set_len(poly_size);
        }
        crate::DeviceMemPool::mem_copy_dtoh(&mut poly_values, &gpu_h_poly);
        crate::DeviceMemPool::deallocate(gpu_h_poly);
        crate::DeviceMemPool::deallocate(gpu_h_pieces);
        let h_poly: Polynomial<F, Coeff>
            = Polynomial {
                values: poly_values,
                _marker: PhantomData,
            };

        let mut gpu_poly = self.committed.random_poly_gpu;
        let mut gpu_eval_res = crate::DeviceMemPool::allocate::<F>(1); 
        crate::gpu_eval_polynomial(&gpu_poly, &x, &mut gpu_eval_res);
        let mut random_eval_values = [F::ZERO; 1];
        crate::DeviceMemPool::mem_copy_dtoh(&mut random_eval_values, &gpu_eval_res);
        let random_eval = random_eval_values[0];
        crate::DeviceMemPool::deallocate(gpu_poly);
        crate::DeviceMemPool::deallocate(gpu_eval_res);
        transcript.write(&random_eval)?;

        Ok(Evaluated {
            h_poly,
            committed: self.committed,
        })
    }
}

impl<F: PrimeField> Evaluated<F> {
    pub(crate) fn open(&self, x: F) -> impl Iterator<Item = ProverQuery<'_, F>> + Clone {
        iter::empty()
            .chain(Some(ProverQuery {
                point: x,
                poly: &self.h_poly,
            }))
            .chain(Some(ProverQuery {
                point: x,
                poly: &self.committed.random_poly,
            }))
    }
}
