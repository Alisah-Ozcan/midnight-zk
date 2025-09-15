//! # midnight_proofs

#![cfg_attr(docsrs, feature(doc_cfg))]
// The actual lints we want to disable.
#![allow(clippy::op_ref, clippy::many_single_char_names)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
//#![deny(missing_docs)]
#![deny(unsafe_code)]

pub mod circuit;
pub use halo2curves;
pub mod plonk;
pub mod poly;
pub mod transcript;

pub mod dev;
pub mod utils;
 
use std::ffi::CStr;
use std::os::raw::{c_char};
use std::fmt;
use sppark::{NTTInputOutputOrder, NTTDirection, NTTType};
use core::ffi::{c_void, c_int};
use group::Group;
use group::prime::PrimeCurveAffine;
use halo2curves::CurveAffine;
pub type GpuPtr = u64;

// ---------------- DeviceMemPool ----------------

extern "C" {
    fn gpu_set_device(device_index : usize)-> sppark::Error;

    fn gpu_memory_pool_initialize() -> sppark::Error;
    fn gpu_memory_pool_enable() -> sppark::Error;
    fn gpu_memory_pool_disable() -> sppark::Error;
    fn gpu_memory_pool_allocate(size : usize, pointer: *mut GpuPtr) -> sppark::Error;
    fn gpu_memory_pool_deallocate(size : usize, pointer: *mut GpuPtr) -> sppark::Error;

    fn gpu_memory_transfer_host_to_device(dst: *mut GpuPtr, src: *const c_void, bytes: usize) -> sppark::Error;
    fn gpu_memory_transfer_device_to_host(dst: *mut c_void, src: *const GpuPtr, bytes: usize) -> sppark::Error;
    fn gpu_memory_transfer_device_to_device(dst: *mut GpuPtr, src: *const GpuPtr, bytes: usize) -> sppark::Error;
}

#[derive(Copy, Clone, Debug)]
pub struct GpuVec {
    pub addr: u64,         // pointer (adress)
    pub size_bytes: usize, // total size (byte)
    pub elem_size: usize,  // size per element (byte)
}

impl GpuVec {
    pub fn len(&self) -> usize { self.size_bytes / self.elem_size }
    pub fn is_null(&self) -> bool { self.addr == 0 }
}

#[allow(unsafe_code)]
pub fn set_gpu(device_index : usize) {
    let error = unsafe {gpu_set_device(device_index)};

    if error.code != 0 {
        panic!("{}", String::from(error));
    }
}

#[derive(Debug)]
pub struct DeviceMemPool;

impl DeviceMemPool {
    #[allow(unsafe_code)]
    pub fn initialize(device_index : usize)  {
        let error = unsafe {gpu_set_device(device_index);
                gpu_memory_pool_initialize()};

        if error.code != 0 {
            panic!("{}", String::from(error));
        }
    }

    #[allow(unsafe_code)]
    pub fn enable()  {
        let error = unsafe {gpu_memory_pool_enable()};

        if error.code != 0 {
            panic!("{}", String::from(error));
        }
    }

    #[allow(unsafe_code)]
    pub fn disable() {
        let error = unsafe {gpu_memory_pool_disable()};

        if error.code != 0 {
            panic!("{}", String::from(error));
        }
    }

    #[allow(unsafe_code)]
    pub fn allocate<T>(size : usize) -> GpuVec {
        let elem = std::mem::size_of::<T>();
        let bytes = elem * size;

        let mut addr: GpuPtr = 0;
        let error = unsafe { gpu_memory_pool_allocate(bytes, &mut addr) };

        if error.code != 0 {
            panic!("{}", String::from(error));
        }

        GpuVec {
            addr,
            size_bytes: bytes,
            elem_size: elem,
        }
    }

    #[allow(unsafe_code)]
    pub fn deallocate(input: GpuVec)   {
        let mut addr = input.addr;
        let error = unsafe { gpu_memory_pool_deallocate(input.size_bytes, &mut addr) };
            
        if error.code != 0 {
            panic!("{}", String::from(error));
        }
    }

    #[allow(unsafe_code)]
    pub fn mem_copy_htod<T>(dest: &mut GpuVec, src: &[T])   {
        assert!(dest.size_bytes == (std::mem::size_of::<T>() * src.len()));
        let error = unsafe { gpu_memory_transfer_host_to_device(&mut dest.addr, src.as_ptr() as *const c_void, dest.size_bytes) };

        if error.code != 0 {
            panic!("{}", String::from(error));
        }
    }

    #[allow(unsafe_code)]
    pub fn mem_copy_dtoh<T>(dest: &mut [T], src: &GpuVec)   {
        assert!(src.size_bytes == (std::mem::size_of::<T>() * dest.len()));
        let error = unsafe { gpu_memory_transfer_device_to_host(dest.as_mut_ptr() as *mut _, &src.addr, src.size_bytes) };

        if error.code != 0 {
            panic!("{}", String::from(error));
        }
    }

    #[allow(unsafe_code)]
    pub fn mem_copy_dtod(dest: &mut GpuVec, src: &GpuVec)   {
        assert!(src.size_bytes == dest.size_bytes);
        let error = unsafe { gpu_memory_transfer_device_to_device(&mut dest.addr, &src.addr, src.size_bytes) };

        if error.code != 0 {
            panic!("{}", String::from(error));
        }
    }
}

extern "C" {
    fn gpu_coset_ntt(
        input_ptr: *const GpuPtr, 
        input_len: usize,
        output_ptr: *mut GpuPtr, 
        output_len: usize,
        g_coset: *const c_void,
        g_coset_inv: *const c_void,
    ) -> sppark::Error;
}

#[allow(unsafe_code)]
pub fn gpu_coeff_to_extended<T: std::clone::Clone>(
    dest: &mut GpuVec,
    src: &GpuVec,
    g_coset: &T,
    g_coset_inv: &T
) 
{   
    let g_coset_p = &[ g_coset.clone() ];
    let g_coset_inv_p = &[ g_coset_inv.clone() ];

    unsafe {
        gpu_coset_ntt(
        &src.addr,
        src.len(),
        &mut dest.addr,
        dest.len(),
        g_coset_p.as_ptr() as *const c_void,
        g_coset_inv_p.as_ptr() as *const c_void,
        );
    };
}









extern "C" {
    fn gpu_coset_intt(
        input_ptr: *const GpuPtr, 
        input_len: usize,
        output_ptr: *mut GpuPtr, 
        output_len: usize,
        g_coset: *const c_void,
        g_coset_inv: *const c_void,
    ) -> sppark::Error;
}

#[allow(unsafe_code)]
pub fn gpu_extended_to_coeff<T: std::clone::Clone>(
    dest: &mut GpuVec,
    src: &GpuVec,
    g_coset: &T,
    g_coset_inv: &T
) 
{   
    let g_coset_p = &[ g_coset.clone() ];
    let g_coset_inv_p = &[ g_coset_inv.clone() ];

    unsafe {
        gpu_coset_intt(
        &src.addr,
        src.len(),
        &mut dest.addr,
        dest.len(),
        g_coset_p.as_ptr() as *const c_void,
        g_coset_inv_p.as_ptr() as *const c_void,
        );
    };
}

extern "C" {
    fn gpu_intt(
        input_ptr: *const GpuPtr, 
        input_len: usize,
    ) -> sppark::Error;
}

#[allow(unsafe_code)]
pub fn gpu_lagrange_to_coeff<T: std::clone::Clone>(
    input: &mut GpuVec,
) 
{   
    unsafe {
        gpu_intt(
        &input.addr,
        input.len()
        );
    }
}

extern "C" {
    fn gpu_msm(
        out1: *mut c_void,
        out2: *mut c_void,
        points_with_infinity: *const c_void,
        npoints: usize,
        scalars: *const GpuPtr,
        ffi_affine_sz: usize,
    ) -> sppark::Error;
}

#[allow(unsafe_code)]
/// Perform MSM GPU
pub fn msm_gpu<C>(points: &[C::Curve], scalars: &GpuVec) -> C::Curve
where
    C: PrimeCurveAffine,
{
    let npoints = points.len();

    if npoints != scalars.len() {
        panic!("length mismatch")
    }

    let mut ret = C::Curve::identity();
    let mut ret2 = C::Curve::identity();
    let err = unsafe {
        gpu_msm(
            &mut ret as *mut _ as *mut _,
            &mut ret2 as *mut _ as *mut _,
            points.as_ptr() as *const _,
            npoints,
            &scalars.addr,
            std::mem::size_of::<C::Curve>(),
        )
    };

    if err.code != 0 {
        panic!("MSM GPU error: {}", String::from(err));
    }

    ret + ret2
}


extern "C" {
    fn gpu_msm_lagrange(
        out1: *mut c_void,
        out2: *mut c_void,
        points_with_infinity: *const c_void,
        npoints: usize,
        scalars: *const GpuPtr,
        ffi_affine_sz: usize,
    ) -> sppark::Error;
}

#[allow(unsafe_code)]
pub fn msm_gpu_lagrange<C>(points: &[C::Curve], scalars: &GpuVec) -> C::Curve
where
    C: PrimeCurveAffine,
{
    let npoints = points.len();

    if npoints != scalars.len() {
        panic!("length mismatch")
    }

    let mut ret = C::Curve::identity();
    let mut ret2 = C::Curve::identity();
    let err = unsafe {
        gpu_msm_lagrange(
            &mut ret as *mut _ as *mut _,
            &mut ret2 as *mut _ as *mut _,
            points.as_ptr() as *const _,
            npoints,
            &scalars.addr,
            std::mem::size_of::<C::Curve>(), // std::mem::size_of::<C>()
        )
    };

    if err.code != 0 {
        panic!("MSM GPU error: {}", String::from(err));
    }

    ret + ret2
}







extern "C" {
    fn gpu_divide_by_vanishing_poly_ptr_(
        in1_gpu_ptr: GpuPtr,
        in2: *const core::ffi::c_void,
        npoints: usize,
        mask: usize
    ) -> sppark::Error;
}

#[allow(unsafe_code)]
pub fn gpu_divide_by_vanishing_poly_ptr<T>(
    in1: &GpuVec, 
    in2: &[T],
    mask: usize
) {
    let npoints = in1.len();
    if (npoints & (npoints - 1)) != 0 {
        panic!("npoints is not power of 2");
    }
    if (mask & (mask + 1)) != 0 {
        panic!("(mask + 1) is not power of 2");
    }

    let err = unsafe {
        gpu_divide_by_vanishing_poly_ptr_(
            in1.addr,
            in2.as_ptr() as *const _,
            npoints,
            mask
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}



extern "C" {
    fn gpu_eval_poly(
        in_ptr: GpuPtr,
        npoints: usize,
        base: *const core::ffi::c_void,
        out_ptr: GpuPtr,
    ) -> sppark::Error;
}

/// Evaluate a polynomial (given in coefficient form) at a field element `base` on the GPU.
/// The input `poly` is a `GpuVec` of coefficients a_0 .. a_{n-1}. The result is written
/// into `out` (which must have at least 1 field element of allocated space on device).
/// Returns nothing; panics on CUDA error.
#[allow(unsafe_code)]
pub fn gpu_eval_polynomial<F: Clone>(poly: &GpuVec, base: &F, out: &GpuVec) {
    if out.len() == 0 {
        panic!("output GpuVec must have space for result (len >= 1)");
    }
    let npoints = poly.len();
    if npoints == 0 { panic!("polynomial length must be > 0"); }
    let err = unsafe { gpu_eval_poly(poly.addr, npoints, base as *const F as *const core::ffi::c_void, out.addr) };
    if err.code != 0 { panic!("{}", String::from(err)); }
}

extern "C" {
    fn gpu_mul_zetas_ptr_(
        in_ptr: GpuPtr,
        npoints: usize,
        zeta: *const core::ffi::c_void,
        zeta_inv: *const core::ffi::c_void,
    ) -> sppark::Error;
}

/// Multiply GPU polynomial by powers of zeta (coset transformation)
#[allow(unsafe_code)]
pub fn gpu_mul_zetas_ptr<F>(
    gpu_ptr: &GpuVec,
    zeta: F,
    zeta_inv: F
) {
    let err = unsafe {
        gpu_mul_zetas_ptr_(
            gpu_ptr.addr,
            gpu_ptr.len(),
            &zeta as *const F as *const core::ffi::c_void, //  Pass address of zeta
            &zeta_inv as *const F as *const core::ffi::c_void, //  Pass address of zeta_inv
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}


extern "C" {
    fn sppark_ntt_ptr(
        device_id: usize,
        inout: GpuPtr,
        lg_domain_size: u32,
        ntt_order: NTTInputOutputOrder,
        ntt_direction: NTTDirection,
        ntt_type: NTTType,
    ) -> sppark::Error;
}

/// Compute an in-place forward NTT on the input data.
#[allow(unsafe_code)]
pub fn intt_gpu_ptr(device_id: usize, inout: &GpuVec, order: NTTInputOutputOrder) {
    if (inout.len() & (inout.len() - 1)) != 0 {
        panic!("inout.len() is not power of 2");
    }

    let err = unsafe {
        sppark_ntt_ptr(
            device_id,
            inout.addr,
            inout.len().trailing_zeros(),
            order,
            NTTDirection::Inverse,
            NTTType::Standard,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}


///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum ValueSourceKind {
    Constant,
    Intermediate,
    Fixed,
    Advice,
    Instance,
    Challenge,
    Beta,
    Gamma,
    Theta,
    TrashChallenge,
    Y,
    PreviousValue,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ValueSourceFFI {
    pub kind: ValueSourceKind,
    pub param0: usize,
    pub param1: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum CalculationKind {
    Add,
    Sub,
    Mul,
    Square,
    Double,
    Negate,
    Store,
    Horner,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct CalculationFFI {
    pub kind: CalculationKind,
    pub a: ValueSourceFFI,
    pub b: ValueSourceFFI,
    pub extra: ValueSourceFFI,
    pub horner_parts_ptr: *const ValueSourceFFI,
    pub horner_parts_len: usize,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct CalculationInfoFFI {
    pub calculation: CalculationFFI,
    pub target: usize,
}

extern "C" {
    fn custom_gates_evaluation(
        calculations: *const CalculationInfoFFI,
        calculations_count: usize,

        fixed_ptrs: *const GpuPtr,

        advice_ptrs: *const GpuPtr,

        instance_ptrs: *const GpuPtr,

        challenges: *const c_void,
        challenges_ptr_len: usize,

        beta: *const c_void,
        gamma: *const c_void,
        theta: *const c_void,
        trash_challenge: *const c_void,
        y: *const c_void,

        output: *mut GpuPtr,

        constants: *const c_void,
        constants_ptr_len: usize,

        rotation_value: *const c_int,
        rotation_ptr_len: usize,

        rot_scale: c_int,
        poly_size: c_int,
    ) -> sppark::Error;
}

#[allow(unsafe_code)]
pub fn custom_gates_evaluation_r<T: std::clone::Clone>(
    calculation: &[CalculationInfoFFI],
    fixed_ptrs: &[u64],
    advice_ptrs: &[u64],
    instance_ptrs: &[u64],
    challenges: &[T],
    beta: &T, gamma: &T, theta: &T, trash_challenge: &T, y: &T, 
    output:  &mut GpuVec,
    constants: &[T],
    rotation_value: &Vec<i32>,
    rot_scale: &i32,
    poly_size: &i32,
) 
{ 
    let beta_p = &[ beta.clone() ];
    let gamma_p = &[ gamma.clone() ];
    let theta_p = &[ theta.clone() ];
    let trash_challenge_p = &[ trash_challenge.clone() ];
    let y_p = &[ y.clone() ];

    unsafe {
        custom_gates_evaluation(calculation.as_ptr(), calculation.len(),
        fixed_ptrs.as_ptr(),
        advice_ptrs.as_ptr(),
        instance_ptrs.as_ptr(),
        challenges.as_ptr() as *const c_void, challenges.len(),

        beta_p.as_ptr() as *const c_void,
        gamma_p.as_ptr() as *const c_void,
        theta_p.as_ptr() as *const c_void,
        trash_challenge_p.as_ptr() as *const c_void,
        y_p.as_ptr() as *const c_void,

        &mut output.addr,

        constants.as_ptr() as *const c_void, constants.len(),

        rotation_value.as_ptr(), rotation_value.len(),

        *rot_scale, *poly_size
        );
    }

}


#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum AnyFFI {
    Advice,
    Fixed,
    Instance,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct ColumnFFI {
    pub index: usize,
    pub column_type: AnyFFI,
}

extern "C" {
    fn permutations_evaluation(
        column: *const ColumnFFI,
        column_count: usize,

        fixed_ptrs: *const GpuPtr,

        advice_ptrs: *const GpuPtr,

        instance_ptrs: *const GpuPtr,

        value: *mut GpuPtr,

        l0_ptrs: *const GpuPtr,
        l_last_ptrs: *const GpuPtr,
        l_active_row_ptrs: *const GpuPtr,
        pk_coset_ptrs: *const GpuPtr,

        permutation_ptrs: *const GpuPtr,
        permutation_ptr_len: usize,


        delta_start: *const c_void,
        delta: *const c_void,
        beta: *const c_void,
        gamma: *const c_void,
        y: *const c_void,
        extended_omega: *const c_void,
        
        chunk_len: c_int,
        last_rotation_value: c_int,

        rot_scale: c_int,
        poly_size: c_int
    ) -> sppark::Error;
}


#[allow(unsafe_code)]
pub fn permutations_evaluation_r<T: std::clone::Clone>(
    column: &[ColumnFFI],

    fixed_ptrs: &[u64],
    advice_ptrs: &[u64],
    instance_ptrs: &[u64],

    value:  &mut GpuVec,

    l0_ptrs: &GpuVec,
    l_last_ptrs: &GpuVec,
    l_active_row_ptrs: &GpuVec,

    pk_coset_ptrs: &[u64],
    permutation_ptrs: &[u64],
    
    delta_start: &T, delta: &T,

    beta: &T, gamma: &T, y: &T, extended_omega: &T, 

    chunk_len: &i32,
    last_rotation_value: &i32,
    rot_scale: &i32,
    poly_size: &i32,
) 
{   
    let delta_start_p = &[ delta_start.clone() ];
    let delta_p = &[ delta.clone() ];
    let beta_p = &[ beta.clone() ];
    let gamma_p = &[ gamma.clone() ];
    let y_p = &[ y.clone() ];
    let extended_omega_p = &[ extended_omega.clone() ];

    unsafe {
        permutations_evaluation(column.as_ptr(), column.len(),
        fixed_ptrs.as_ptr(),
        advice_ptrs.as_ptr(),
        instance_ptrs.as_ptr(),

        &mut value.addr,

        &l0_ptrs.addr,
        &l_last_ptrs.addr,
        &l_active_row_ptrs.addr,

        pk_coset_ptrs.as_ptr(),
        permutation_ptrs.as_ptr(),
        permutation_ptrs.len(),

        delta_start_p.as_ptr() as *const c_void,
        delta_p.as_ptr() as *const c_void,
        beta_p.as_ptr() as *const c_void,
        gamma_p.as_ptr() as *const c_void,
        y_p.as_ptr() as *const c_void,
        extended_omega_p.as_ptr() as *const c_void,

        *chunk_len, *last_rotation_value,
        *rot_scale, *poly_size,
        );
    }
}

extern "C" {
    fn lookups_evaluation(
        calculations: *const CalculationInfoFFI,
        calculations_count: usize,

        fixed_ptrs: *const GpuPtr,
        advice_ptrs: *const GpuPtr,
        instance_ptrs: *const GpuPtr,

        l0_ptrs: *const GpuPtr,
        l_last_ptrs: *const GpuPtr,
        l_active_row_ptrs: *const GpuPtr,

        value: *mut GpuPtr,

        product_coset_ptrs: *const GpuPtr,
        permuted_input_coset: *const GpuPtr,
        permuted_table_coset: *const GpuPtr,

        challenges: *const c_void,
        challenges_ptr_len: usize,

        beta: *const c_void,
        gamma: *const c_void,
        theta: *const c_void,
        trash_challenge: *const c_void,
        y: *const c_void,

        constants: *const c_void,
        constants_ptr_len: usize,

        rotation_value: *const c_int,
        rotation_ptr_len: usize,

        rot_scale: c_int,
        poly_size: c_int,
    ) -> sppark::Error;
}

#[allow(unsafe_code)]
pub fn lookups_evaluation_r<T: std::clone::Clone>(
    calculation: &[CalculationInfoFFI],

    fixed_ptrs: &[u64],
    advice_ptrs: &[u64],
    instance_ptrs: &[u64],

    l0_ptrs: &GpuVec,
    l_last_ptrs: &GpuVec,
    l_active_row_ptrs: &GpuVec,

    value:  &mut GpuVec,

    product_coset_ptrs: &GpuVec,
    permuted_input_coset: &GpuVec,
    permuted_table_coset: &GpuVec,

    challenges: &[T],

    beta: &T, gamma: &T, theta: &T, trash_challenge: &T, y: &T, 
    constants: &[T],
    rotation_value: &Vec<i32>,
    rot_scale: &i32,
    poly_size: &i32,
) 
{ 
    let beta_p = &[ beta.clone() ];
    let gamma_p = &[ gamma.clone() ];
    let theta_p = &[ theta.clone() ];
    let trash_challenge_p = &[ trash_challenge.clone() ];
    let y_p = &[ y.clone() ];

    unsafe {
        lookups_evaluation(calculation.as_ptr(), calculation.len(),
        fixed_ptrs.as_ptr(),
        advice_ptrs.as_ptr(),
        instance_ptrs.as_ptr(),

        &l0_ptrs.addr,
        &l_last_ptrs.addr,
        &l_active_row_ptrs.addr,

        &mut value.addr,

        &product_coset_ptrs.addr,
        &permuted_input_coset.addr,
        &permuted_table_coset.addr,

        challenges.as_ptr() as *const c_void, challenges.len(),

        beta_p.as_ptr() as *const c_void,
        gamma_p.as_ptr() as *const c_void,
        theta_p.as_ptr() as *const c_void,
        trash_challenge_p.as_ptr() as *const c_void,
        y_p.as_ptr() as *const c_void,

        constants.as_ptr() as *const c_void, constants.len(),

        rotation_value.as_ptr(), rotation_value.len(),

        *rot_scale, *poly_size
        );
    }
}

extern "C" {
    fn commit_product(
        permuted_input_value_device_ptrs: *const GpuPtr,
        permuted_table_value_device_ptrs: *const GpuPtr,
        compressed_input_expression_device_ptrs: *const GpuPtr,
        compressed_table_expression_device_ptrs: *const GpuPtr,

        value: *mut GpuPtr,

        beta: *const c_void,
        gamma: *const c_void,
        randoms: *const c_void,
        one: *const c_void,
        blinding_factors: c_int,
        isize: c_int
    ) -> sppark::Error;
}


#[allow(unsafe_code)]
pub fn commit_product_r<T: std::clone::Clone>(
    permuted_input_value: &GpuVec,
    permuted_table_value: &GpuVec,
    compressed_input_expression: &GpuVec,
    compressed_table_expression: &GpuVec,
    value:  &mut GpuVec,

    beta: &T,
    gamma: &T,
    randoms: &[T],
    one: &T,
    blinding_factors: &i32,
    isize: &i32
)   
{   
    let one_p = &[ one.clone() ];
    let beta_p = &[ beta.clone() ];
    let gamma_p = &[ gamma.clone() ];

    unsafe {
        commit_product(
        &permuted_input_value.addr,
        &permuted_table_value.addr,
        &compressed_input_expression.addr,
        &compressed_table_expression.addr,

        &mut value.addr,

        beta_p.as_ptr() as *const c_void, 
        gamma_p.as_ptr() as *const c_void, 

        randoms.as_ptr() as *const c_void, 
        one_p.as_ptr() as *const c_void, 
        *blinding_factors,
        *isize,
        );
    }
}