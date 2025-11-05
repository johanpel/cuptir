//! Random crate utilities
use std::{ffi::CStr, ptr::NonNull};

/// Return a CStr from a C string if the C string is not a nullptr.
///
/// # Safety
/// Other safety rules are described by [CStr::from_ptr]
pub(crate) unsafe fn try_cstr_from_ffi<'a>(c_str: *const std::ffi::c_char) -> Option<&'a CStr> {
    NonNull::new(c_str as *mut _).map(|p| unsafe { CStr::from_ptr(p.as_ptr()) })
}

/// Return a &str from a C string if the C string is not a nullptr and is valid UTF-8.
///
/// # Safety
/// Other safety rules are described by [CStr::from_ptr]
pub(crate) unsafe fn try_str_from_ffi<'a>(c_str: *const std::ffi::c_char) -> Option<&'a str> {
    unsafe { try_cstr_from_ffi(c_str) }.and_then(|c_str| c_str.to_str().ok())
}

/// Return a demangled symbol name if C string is not a nullptr and represents a symbol
/// name.
///
/// # Safety
/// Other safety rules are described by [CStr::from_ptr]
pub(crate) unsafe fn try_demangle_from_ffi(c_str: *const std::ffi::c_char) -> Option<String> {
    unsafe { try_cstr_from_ffi(c_str) }.and_then(|c_str| {
        c_str.to_str().ok().map(|utf8_str| {
            cpp_demangle::Symbol::new(utf8_str)
                .ok()
                .map(|symbol| symbol.demangle().ok())
                .flatten()
                .unwrap_or(utf8_str.to_owned())
        })
    })
}

/// Return a UUID constructed from a slice of c_chars.
pub(crate) fn uuid_from_i8_slice(bytes: [std::ffi::c_char; 16]) -> uuid::Uuid {
    uuid::Uuid::from_bytes(bytes.map(|x| x as u8))
}
