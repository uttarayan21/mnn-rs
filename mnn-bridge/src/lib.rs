#[cfg(not(any(
     all(feature = "ndarray_0_15", not(any(feature = "ndarray_0_16", feature = "ndarray_0_17"))),
     all(feature = "ndarray_0_16", not(any(feature = "ndarray_0_15", feature = "ndarray_0_17"))),
     all(feature = "ndarray_0_17", not(any(feature = "ndarray_0_15", feature = "ndarray_0_16"))),
)))]
compile_error!("Only one of `ndarray_0_15`, `ndarray_0_16`, `ndarray` (`ndarray_0_17`) must be enabled");


#[cfg(feature = "ndarray_0_17")]
pub mod ndarray_0_17 {
    use ndarray_0_17 as ndarray;
    include!("ndarray.rs");
}

#[cfg(feature = "ndarray_0_16")]
pub mod ndarray_0_16 {
    use ndarray_0_16 as ndarray;
    include!("ndarray.rs");
}
#[cfg(feature = "ndarray_0_15")]
mod ndarray_0_15 {
    use ndarray_0_15 as ndarray;
    include!("ndarray.rs");
}
