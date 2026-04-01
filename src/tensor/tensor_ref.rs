use super::*;

/// This is modeled as a reference to a tensor, you should never be able to dereference this.
/// At any point you should only have &TensorRef or &mut TensorRef, never an owned TensorRef.
pub struct TensorRef<H, M>
where
    H: HalideType,
    M: TensorMachine,
{
    __marker: PhantomData<(M, mnn_sys::Tensor, [H])>,
}

impl<H, M> core::fmt::Debug for TensorRef<H, M>
where
    H: HalideType,
    M: TensorMachine,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorRef")
            .field("shape", &self.shape())
            .field("device_id", &self.device_id())
            .field("dimensions", &self.get_dimension_type())
            .field("elements", &self.element_size())
            .field("dynamic", &self.is_dynamic_unsized())
            .field("size", &self.size())
            .finish()
    }
}

impl<H: HalideType, M: TensorMachine> TensorRef<H, M> {
    /// Get a raw pointer to the underlying MNN tensor
    pub(crate) fn as_ptr(&self) -> *mut mnn_sys::Tensor {
        self as *const Self as *mut mnn_sys::Tensor
    }

    /// Copies the data from a host tensor to the self.as_ptr()
    pub fn copy_from_host_tensor(&mut self, tensor: &TensorRef<H, Host>) -> Result<()> {
        assert_eq!(self.size(), tensor.size(), "Tensor sizes do not match");
        let ret = unsafe { Tensor_copyFromHostTensor(self.as_ptr(), tensor.as_ptr()) };
        crate::ensure!(ret != 0, ErrorKind::TensorCopyFailed(ret));
        Ok(())
    }

    /// Copies the data from the self.as_ptr() to a host tensor
    pub fn copy_to_host_tensor(&self, tensor: &mut TensorRef<H, Host>) -> Result<()> {
        assert_eq!(self.size(), tensor.size(), "Tensor sizes do not match");
        let ret = unsafe { Tensor_copyToHostTensor(self.as_ptr(), tensor.as_ptr()) };
        crate::ensure!(ret != 0, ErrorKind::TensorCopyFailed(ret));
        Ok(())
    }
}

impl<H, M> core::ops::Deref for TensorRef<H, M>
where
    H: HalideType,
    M: TensorMachine,
{
    type Target = AnyTensorRef;
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self.as_ptr() as *const AnyTensorRef) }
    }
}

impl<H, M> core::ops::DerefMut for TensorRef<H, M>
where
    H: HalideType,
    M: TensorMachine,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self.as_ptr().cast()) }
    }
}

impl<T> TensorRef<T, Device>
where
    T: HalideType,
{
    /// Try to wait for the device tensor to finish processing
    pub fn wait(&self, map_type: MapType, finish: bool) {
        unsafe {
            Tensor_wait(self.as_ptr(), map_type, finish as i32);
        }
    }

    /// Create a host tensor from the device tensor with same dimensions and data type and
    /// optionally copy the data from the device tensor
    pub fn create_host_tensor_from_device(&self, copy_data: bool) -> Tensor<Owned<T>, Host> {
        let shape = self.shape();
        let dm_type = self.get_dimension_type();
        let mut out = Tensor::new(shape, dm_type);

        if copy_data {
            self.copy_to_host_tensor(&mut out)
                .expect("Failed to copy data from device tensor");
        }
        out
    }
}

/// Construct a tensor reference from a raw pointer to an MNN tensor
///
/// # Safety
/// The caller must ensure that the provided pointer is valid and points to a properly initialized MNN
#[inline]
pub unsafe fn from_raw_parts<'a, H, M>(tensor: *const mnn_sys::Tensor) -> &'a TensorRef<H, M>
where
    H: HalideType,
    M: TensorMachine,
{
    unsafe { &*tensor.cast::<TensorRef<H, M>>() }
}

/// Construct a mutable tensor reference from a raw pointer to an MNN tensor
///
/// # Safety
/// The caller must ensure that the provided pointer is valid and points to a properly initialized
#[inline]
pub unsafe fn from_raw_parts_mut<'a, H, M>(tensor: *mut mnn_sys::Tensor) -> &'a mut TensorRef<H, M>
where
    H: HalideType,
    M: TensorMachine,
{
    unsafe { &mut *tensor.cast::<TensorRef<H, M>>() }
}

impl<S, M, H> core::ops::Deref for Tensor<S, M, H>
where
    S: TensorType<H = H>,
    H: HalideType,
    M: TensorMachine,
{
    type Target = TensorRef<H, M>;

    fn deref(&self) -> &Self::Target {
        unsafe { from_raw_parts(self.tensor) }
    }
}

impl<S, M, H> core::ops::DerefMut for Tensor<S, M, H>
where
    S: TensorType<H = H>,
    H: HalideType,
    M: TensorMachine,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { from_raw_parts_mut(self.tensor) }
    }
}

impl<T, M> TensorRef<T, M>
where
    T: HalideType,
    M: TensorMachine,
{
    /// Fill the tensor with the specified value
    pub fn fill(&mut self, value: T) -> Result<()>
    where
        T: Copy,
    {
        if !self.is_type_of::<T>() {
            unimplemented!(
                "Filling tensor of type {:?} with value of type {:?} is not supported",
                self.get_dimension_type(),
                halide_type_of::<T>()
            );
        }
        if M::host() {
            let size = self.element_size();
            let result: &mut [T] = unsafe {
                let data = mnn_sys::Tensor_host_mut(self.as_ptr()).cast();
                core::slice::from_raw_parts_mut(data, size)
            };
            result.fill(value);
        } else if M::device() {
            let shape = self.shape();
            let dm_type = self.get_dimension_type();
            let mut host = Tensor::new(shape, dm_type);
            host.fill(value)?;
            self.copy_from_host_tensor(&host)?;
        }
        Ok(())
    }
}

impl<T> TensorRef<T, Host>
where
    T: HalideType,
{
    /// Try to map the device tensor to the host memory and get the slice
    pub fn try_host(&self) -> Result<&[T]> {
        let size = self.element_size();
        ensure!(
            self.is_type_of::<T>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<T>(),
            }
        );
        let result = unsafe {
            let data = mnn_sys::Tensor_host(self.as_ptr()).cast();
            core::slice::from_raw_parts(data, size)
        };
        Ok(result)
    }

    /// Try to map the device tensor to the host memory and get the mutable slice
    pub fn try_host_mut(&mut self) -> Result<&mut [T]> {
        let size = self.element_size();
        ensure!(
            self.is_type_of::<T>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<T>(),
            }
        );

        let result = unsafe {
            let data: *mut T = mnn_sys::Tensor_host_mut(self.as_ptr()).cast();
            debug_assert!(!data.is_null());
            core::slice::from_raw_parts_mut(data, size)
        };
        Ok(result)
    }

    /// Get the host memory slice of the tensor
    pub fn host(&self) -> &[T] {
        self.try_host().expect("Failed to get tensor host")
    }

    /// Get the mutable host memory slice of the tensor
    pub fn host_mut(&mut self) -> &mut [T] {
        self.try_host_mut().expect("Failed to get tensor host_mut")
    }
}
