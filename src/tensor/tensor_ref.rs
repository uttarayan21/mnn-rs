use super::*;

/// A reference to a tensor
/// This is analogous to &[T] in Rust, which is a reference to a slice of T
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
            // .field("len", &self.len())
            .finish()
    }
}

impl<H: HalideType, M: TensorMachine> TensorRef<H, M> {
    /// Get a raw pointer to the underlying MNN tensor
    pub fn as_ptr(&self) -> *mut mnn_sys::Tensor {
        unsafe { core::mem::transmute::<&Self, *mut mnn_sys::Tensor>(self) }
    }

    /// Get the device id of the tensor
    pub fn device_id(&self) -> u64 {
        unsafe { Tensor_deviceId(self.as_ptr()) }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> TensorShape {
        unsafe { Tensor_shape(self.as_ptr()) }.into()
    }

    /// Get the dimensions of the tensor
    #[doc(alias = "dims")]
    pub fn dimensions(&self) -> usize {
        unsafe { Tensor_dimensions(self.as_ptr()) as usize }
    }

    /// Get the width of the tensor
    pub fn width(&self) -> u32 {
        unsafe { Tensor_width(self.as_ptr()) as u32 }
    }

    /// Get the height of the tensor
    pub fn height(&self) -> u32 {
        unsafe { Tensor_height(self.as_ptr()) as u32 }
    }

    /// Get the channel size of the tensor
    pub fn channel(&self) -> u32 {
        unsafe { Tensor_channel(self.as_ptr()) as u32 }
    }

    /// Get the batch size of the tensor
    pub fn batch(&self) -> u32 {
        unsafe { Tensor_batch(self.as_ptr()) as u32 }
    }

    /// Get the size of the tensor when counted by bytes
    pub fn size(&self) -> usize {
        unsafe { Tensor_usize(self.as_ptr()) }
    }

    /// Get the size of the tensor when counted by elements
    pub fn element_size(&self) -> usize {
        unsafe { Tensor_elementSize(self.as_ptr()) as usize }
    }

    /// Check if the tensor is of the specified data type
    pub fn is_type_of<Ha: HalideType>(&self) -> bool {
        let htc = halide_type_of::<Ha>();
        unsafe { Tensor_isTypeOf(self.as_ptr(), htc) }
    }

    /// Get the dimension type of the tensor
    pub fn get_dimension_type(&self) -> DimensionType {
        debug_assert!(!self.as_ptr().is_null());
        From::from(unsafe { Tensor_getDimensionType(self.as_ptr()) })
    }

    /// Check if the tensor is dynamic and needs resizing
    pub fn is_dynamic_unsized(&self) -> bool {
        self.shape().as_ref().contains(&-1)
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
    unsafe { core::mem::transmute::<_, &TensorRef<H, M>>(tensor) }
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
    unsafe { core::mem::transmute::<_, &mut TensorRef<H, M>>(tensor) }
}

impl<S, M, H> core::ops::Deref for Tensor<S, M, H>
where
    S: TensorType<H = H>,
    H: HalideType,
    M: TensorMachine,
{
    type Target = TensorRef<H, M>;

    fn deref(&self) -> &Self::Target {
        unsafe { core::mem::transmute::<_, &TensorRef<H, M>>(self.tensor) }
    }
}

impl<S, M, H> core::ops::DerefMut for Tensor<S, M, H>
where
    S: TensorType<H = H>,
    H: HalideType,
    M: TensorMachine,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { core::mem::transmute::<_, &mut TensorRef<H, M>>(self.tensor) }
    }
}
