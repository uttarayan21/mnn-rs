use super::*;
/// This is a zero sized type
#[repr(transparent)]
pub struct AnyTensorRef {
    __marker: PhantomData<[u8]>,
}

impl AnyTensorRef {
    /// Get a raw pointer to the underlying MNN tensor
    pub(crate) fn as_ptr(&self) -> *mut mnn_sys::Tensor {
        self as *const Self as *mut mnn_sys::Tensor
    }

    pub(crate) unsafe fn from_ptr<'s>(tensor: *mut mnn_sys::Tensor) -> &'s Self {
        unsafe { &*tensor.cast() }
    }

    /// Get the device id of the tensor
    pub fn device_id(&self) -> u64 {
        unsafe { mnn_sys::Tensor_deviceId(self.as_ptr()) }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> TensorShape {
        unsafe { mnn_sys::Tensor_shape(self.as_ptr()) }.into()
    }

    /// Get the dimensions of the tensor
    #[doc(alias = "dims")]
    pub fn dimensions(&self) -> usize {
        unsafe { mnn_sys::Tensor_dimensions(self.as_ptr()) as usize }
    }

    /// Get the width of the tensor
    pub fn width(&self) -> u32 {
        unsafe { mnn_sys::Tensor_width(self.as_ptr()) as u32 }
    }

    /// Get the height of the tensor
    pub fn height(&self) -> u32 {
        unsafe { mnn_sys::Tensor_height(self.as_ptr()) as u32 }
    }

    /// Get the channel size of the tensor
    pub fn channel(&self) -> u32 {
        unsafe { mnn_sys::Tensor_channel(self.as_ptr()) as u32 }
    }

    /// Get the batch size of the tensor
    pub fn batch(&self) -> u32 {
        unsafe { mnn_sys::Tensor_batch(self.as_ptr()) as u32 }
    }

    /// Get the size of the tensor when counted by bytes
    pub fn size(&self) -> usize {
        unsafe { mnn_sys::Tensor_usize(self.as_ptr()) }
    }

    /// Get the size of the tensor when counted by elements
    pub fn element_size(&self) -> usize {
        unsafe { mnn_sys::Tensor_elementSize(self.as_ptr()) as usize }
    }

    /// Check if the tensor is of the specified data type
    pub fn is_type_of<Ha: HalideType>(&self) -> bool {
        let htc = mnn_sys::halide_type_of::<Ha>();
        unsafe { mnn_sys::Tensor_isTypeOf(self.as_ptr(), htc) }
    }

    /// Get the dimension type of the tensor
    pub fn get_dimension_type(&self) -> DimensionType {
        debug_assert!(!self.as_ptr().is_null());
        From::from(unsafe { mnn_sys::Tensor_getDimensionType(self.as_ptr()) })
    }

    /// Check if the tensor is dynamic and needs resizing
    pub fn is_dynamic_unsized(&self) -> bool {
        self.shape().as_ref().contains(&-1)
    }

    /// Copies the data from a host tensor to the self.as_ptr()
    pub fn copy_from_host_tensor(&mut self, tensor: &AnyTensorRef) -> Result<()> {
        // assert_eq!(self.size(), tensor.size(), "Tensor sizes do not match");
        crate::ensure!(
            self.size() == tensor.size(),
            ErrorKind::SizeMismatch {
                expected: self.size(),
                got: tensor.size()
            }
        );
        let ret = unsafe { mnn_sys::Tensor_copyFromHostTensor(self.as_ptr(), tensor.as_ptr()) };
        crate::ensure!(ret != 0, ErrorKind::TensorCopyFailed(ret));
        Ok(())
    }

    /// Copies the data from the self.as_ptr() to a host tensor
    pub fn copy_to_host_tensor(&self, tensor: &mut AnyTensorRef) -> Result<()> {
        assert_eq!(self.size(), tensor.size(), "Tensor sizes do not match");
        let ret = unsafe { mnn_sys::Tensor_copyToHostTensor(self.as_ptr(), tensor.as_ptr()) };
        crate::ensure!(ret != 0, ErrorKind::TensorCopyFailed(ret));
        Ok(())
    }

    // /// Create a host tensor from the device tensor with same dimensions and data type and
    // /// optionally copy the data from the device tensor
    // pub fn create_host_tensor_from_device<T: HalideType>(
    //     &self,
    //     copy_data: bool,
    // ) -> Tensor<Owned<T>, Host> {
    //     let shape = self.shape();
    //     let dm_type = self.get_dimension_type();
    //     let mut out = Tensor::new(shape, dm_type);
    //
    //     if copy_data {
    //         self.copy_to_host_tensor(&mut out)
    //             .expect("Failed to copy data from device tensor");
    //     }
    //     out
    // }

    /// Try to wait for the device tensor to finish processing
    pub fn wait(this: &Self, map_type: MapType, finish: bool) {
        unsafe {
            mnn_sys::Tensor_wait(this.as_ptr(), map_type, finish as i32);
        }
    }
}
