//! Autogenerated: 'fiat-crypto/word_by_word_montgomery' --output fiat_crypto_fp128.rs --lang Rust --inline --inline-internal --no-prefix-fiat --public-function-case snake_case --private-function-case camelCase --public-type-case UpperCamelCase fp128 64 '(2^62-2^3+1)*2^66+1' add sub mul opp from_montgomery to_montgomery to_bytes from_bytes
//! curve description: fp128
//! machine_wordsize = 64 (from "64")
//! requested operations: add, sub, mul, opp, from_montgomery, to_montgomery, to_bytes, from_bytes
//! m = 0xffffffffffffffe40000000000000001 (from "(2^62-2^3+1)*2^66+1")
//!
//! NOTE: In addition to the bounds specified above each function, all
//!   functions synthesized for this Montgomery arithmetic require the
//!   input to be strictly less than the prime modulus (m), and also
//!   require the input to be in the unique saturated representation.
//!   All functions also ensure that these two properties are true of
//!   return values.
//!
//! Computed values:
//!   eval z = z[0] + (z[1] << 64)
//!   bytes_eval z = z[0] + (z[1] << 8) + (z[2] << 16) + (z[3] << 24) + (z[4] << 32) + (z[5] << 40) + (z[6] << 48) + (z[7] << 56) + (z[8] << 64) + (z[9] << 72) + (z[10] << 80) + (z[11] << 88) + (z[12] << 96) + (z[13] << 104) + (z[14] << 112) + (z[15] << 120)
//!   twos_complement_eval z = let x1 := z[0] + (z[1] << 64) in
//!                            if x1 & (2^128-1) < 2^127 then x1 & (2^128-1) else (x1 & (2^128-1)) - 2^128

#![allow(unused_parens)]
#![allow(non_camel_case_types)]

/** Fp128U1 represents a byte. */
pub type Fp128U1 = u8;
/** Fp128I1 represents a byte. */
pub type Fp128I1 = i8;
/** Fp128I2 represents a byte. */
pub type Fp128I2 = i8;

/** The type Fp128MontgomeryDomainFieldElement is a field element in the Montgomery domain. */
/** Bounds: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]] */
#[derive(Clone, Copy)]
pub struct Fp128MontgomeryDomainFieldElement(pub [u64; 2]);

impl core::ops::Index<usize> for Fp128MontgomeryDomainFieldElement {
    type Output = u64;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl core::ops::IndexMut<usize> for Fp128MontgomeryDomainFieldElement {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/** The type Fp128NonMontgomeryDomainFieldElement is a field element NOT in the Montgomery domain. */
/** Bounds: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]] */
#[derive(Clone, Copy)]
pub struct Fp128NonMontgomeryDomainFieldElement(pub [u64; 2]);

impl core::ops::Index<usize> for Fp128NonMontgomeryDomainFieldElement {
    type Output = u64;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl core::ops::IndexMut<usize> for Fp128NonMontgomeryDomainFieldElement {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}


/// The function fp128_addcarryx_u64 is an addition with carry.
///
/// Postconditions:
///   out1 = (arg1 + arg2 + arg3) mod 2^64
///   out2 = ⌊(arg1 + arg2 + arg3) / 2^64⌋
///
/// Input Bounds:
///   arg1: [0x0 ~> 0x1]
///   arg2: [0x0 ~> 0xffffffffffffffff]
///   arg3: [0x0 ~> 0xffffffffffffffff]
/// Output Bounds:
///   out1: [0x0 ~> 0xffffffffffffffff]
///   out2: [0x0 ~> 0x1]
#[inline(always)]
pub fn fp128_addcarryx_u64(out1: &mut u64, out2: &mut Fp128U1, arg1: Fp128U1, arg2: u64, arg3: u64) {
  let x1: u128 = (((arg1 as u128) + (arg2 as u128)) + (arg3 as u128));
  let x2: u64 = ((x1 & (0xffffffffffffffff_u64 as u128)) as u64);
  let x3: Fp128U1 = ((x1 >> 64) as Fp128U1);
  *out1 = x2;
  *out2 = x3;
}

/// The function fp128_subborrowx_u64 is a subtraction with borrow.
///
/// Postconditions:
///   out1 = (-arg1 + arg2 + -arg3) mod 2^64
///   out2 = -⌊(-arg1 + arg2 + -arg3) / 2^64⌋
///
/// Input Bounds:
///   arg1: [0x0 ~> 0x1]
///   arg2: [0x0 ~> 0xffffffffffffffff]
///   arg3: [0x0 ~> 0xffffffffffffffff]
/// Output Bounds:
///   out1: [0x0 ~> 0xffffffffffffffff]
///   out2: [0x0 ~> 0x1]
#[inline(always)]
pub fn fp128_subborrowx_u64(out1: &mut u64, out2: &mut Fp128U1, arg1: Fp128U1, arg2: u64, arg3: u64) {
  let x1: i128 = (((arg2 as i128) - (arg1 as i128)) - (arg3 as i128));
  let x2: Fp128I1 = ((x1 >> 64) as Fp128I1);
  let x3: u64 = ((x1 & (0xffffffffffffffff_u64 as i128)) as u64);
  *out1 = x3;
  *out2 = (((0x0_u8 as Fp128I2) - (x2 as Fp128I2)) as Fp128U1);
}

/// The function fp128_mulx_u64 is a multiplication, returning the full double-width result.
///
/// Postconditions:
///   out1 = (arg1 * arg2) mod 2^64
///   out2 = ⌊arg1 * arg2 / 2^64⌋
///
/// Input Bounds:
///   arg1: [0x0 ~> 0xffffffffffffffff]
///   arg2: [0x0 ~> 0xffffffffffffffff]
/// Output Bounds:
///   out1: [0x0 ~> 0xffffffffffffffff]
///   out2: [0x0 ~> 0xffffffffffffffff]
#[inline(always)]
pub fn fp128_mulx_u64(out1: &mut u64, out2: &mut u64, arg1: u64, arg2: u64) {
  let x1: u128 = ((arg1 as u128) * (arg2 as u128));
  let x2: u64 = ((x1 & (0xffffffffffffffff_u64 as u128)) as u64);
  let x3: u64 = ((x1 >> 64) as u64);
  *out1 = x2;
  *out2 = x3;
}

/// The function fp128_cmovznz_u64 is a single-word conditional move.
///
/// Postconditions:
///   out1 = (if arg1 = 0 then arg2 else arg3)
///
/// Input Bounds:
///   arg1: [0x0 ~> 0x1]
///   arg2: [0x0 ~> 0xffffffffffffffff]
///   arg3: [0x0 ~> 0xffffffffffffffff]
/// Output Bounds:
///   out1: [0x0 ~> 0xffffffffffffffff]
#[inline(always)]
pub fn fp128_cmovznz_u64(out1: &mut u64, arg1: Fp128U1, arg2: u64, arg3: u64) {
  let x1: Fp128U1 = (!(!arg1));
  let x2: u64 = ((((((0x0_u8 as Fp128I2) - (x1 as Fp128I2)) as Fp128I1) as i128) & (0xffffffffffffffff_u64 as i128)) as u64);
  let x3: u64 = ((x2 & arg3) | ((!x2) & arg2));
  *out1 = x3;
}

/// The function fp128_add adds two field elements in the Montgomery domain.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
///   0 ≤ eval arg2 < m
/// Postconditions:
///   eval (from_montgomery out1) mod m = (eval (from_montgomery arg1) + eval (from_montgomery arg2)) mod m
///   0 ≤ eval out1 < m
///
#[inline(always)]
pub fn fp128_add(out1: &mut Fp128MontgomeryDomainFieldElement, arg1: &Fp128MontgomeryDomainFieldElement, arg2: &Fp128MontgomeryDomainFieldElement) {
  let mut x1: u64 = 0;
  let mut x2: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x1, &mut x2, 0x0_u8, (arg1[0]), (arg2[0]));
  let mut x3: u64 = 0;
  let mut x4: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x3, &mut x4, x2, (arg1[1]), (arg2[1]));
  let mut x5: u64 = 0;
  let mut x6: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x5, &mut x6, 0x0_u8, x1, (0x1_u8 as u64));
  let mut x7: u64 = 0;
  let mut x8: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x7, &mut x8, x6, x3, 0xffffffffffffffe4_u64);
  let mut x9: u64 = 0;
  let mut x10: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x9, &mut x10, x8, (x4 as u64), (0x0_u8 as u64));
  let mut x11: u64 = 0;
  fp128_cmovznz_u64(&mut x11, x10, x5, x1);
  let mut x12: u64 = 0;
  fp128_cmovznz_u64(&mut x12, x10, x7, x3);
  out1[0] = x11;
  out1[1] = x12;
}

/// The function fp128_sub subtracts two field elements in the Montgomery domain.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
///   0 ≤ eval arg2 < m
/// Postconditions:
///   eval (from_montgomery out1) mod m = (eval (from_montgomery arg1) - eval (from_montgomery arg2)) mod m
///   0 ≤ eval out1 < m
///
#[inline(always)]
pub fn fp128_sub(out1: &mut Fp128MontgomeryDomainFieldElement, arg1: &Fp128MontgomeryDomainFieldElement, arg2: &Fp128MontgomeryDomainFieldElement) {
  let mut x1: u64 = 0;
  let mut x2: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x1, &mut x2, 0x0_u8, (arg1[0]), (arg2[0]));
  let mut x3: u64 = 0;
  let mut x4: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x3, &mut x4, x2, (arg1[1]), (arg2[1]));
  let mut x5: u64 = 0;
  fp128_cmovznz_u64(&mut x5, x4, (0x0_u8 as u64), 0xffffffffffffffff_u64);
  let mut x6: u64 = 0;
  let mut x7: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x6, &mut x7, 0x0_u8, x1, (((x5 & (0x1_u8 as u64)) as Fp128U1) as u64));
  let mut x8: u64 = 0;
  let mut x9: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x8, &mut x9, x7, x3, (x5 & 0xffffffffffffffe4_u64));
  out1[0] = x6;
  out1[1] = x8;
}

/// The function fp128_mul multiplies two field elements in the Montgomery domain.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
///   0 ≤ eval arg2 < m
/// Postconditions:
///   eval (from_montgomery out1) mod m = (eval (from_montgomery arg1) * eval (from_montgomery arg2)) mod m
///   0 ≤ eval out1 < m
///
#[inline(always)]
pub fn fp128_mul(out1: &mut Fp128MontgomeryDomainFieldElement, arg1: &Fp128MontgomeryDomainFieldElement, arg2: &Fp128MontgomeryDomainFieldElement) {
  let x1: u64 = (arg1[1]);
  let x2: u64 = (arg1[0]);
  let mut x3: u64 = 0;
  let mut x4: u64 = 0;
  fp128_mulx_u64(&mut x3, &mut x4, x2, (arg2[1]));
  let mut x5: u64 = 0;
  let mut x6: u64 = 0;
  fp128_mulx_u64(&mut x5, &mut x6, x2, (arg2[0]));
  let mut x7: u64 = 0;
  let mut x8: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x7, &mut x8, 0x0_u8, x6, x3);
  let x9: u64 = ((x8 as u64) + x4);
  let mut x10: u64 = 0;
  let mut x11: u64 = 0;
  fp128_mulx_u64(&mut x10, &mut x11, x5, 0xffffffffffffffff_u64);
  let mut x12: u64 = 0;
  let mut x13: u64 = 0;
  fp128_mulx_u64(&mut x12, &mut x13, x10, 0xffffffffffffffe4_u64);
  let mut x14: u64 = 0;
  let mut x15: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x14, &mut x15, 0x0_u8, x5, x10);
  let mut x16: u64 = 0;
  let mut x17: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x16, &mut x17, x15, x7, x12);
  let mut x18: u64 = 0;
  let mut x19: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x18, &mut x19, x17, x9, x13);
  let mut x20: u64 = 0;
  let mut x21: u64 = 0;
  fp128_mulx_u64(&mut x20, &mut x21, x1, (arg2[1]));
  let mut x22: u64 = 0;
  let mut x23: u64 = 0;
  fp128_mulx_u64(&mut x22, &mut x23, x1, (arg2[0]));
  let mut x24: u64 = 0;
  let mut x25: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x24, &mut x25, 0x0_u8, x23, x20);
  let x26: u64 = ((x25 as u64) + x21);
  let mut x27: u64 = 0;
  let mut x28: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x27, &mut x28, 0x0_u8, x16, x22);
  let mut x29: u64 = 0;
  let mut x30: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x29, &mut x30, x28, x18, x24);
  let mut x31: u64 = 0;
  let mut x32: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x31, &mut x32, x30, (x19 as u64), x26);
  let mut x33: u64 = 0;
  let mut x34: u64 = 0;
  fp128_mulx_u64(&mut x33, &mut x34, x27, 0xffffffffffffffff_u64);
  let mut x35: u64 = 0;
  let mut x36: u64 = 0;
  fp128_mulx_u64(&mut x35, &mut x36, x33, 0xffffffffffffffe4_u64);
  let mut x37: u64 = 0;
  let mut x38: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x37, &mut x38, 0x0_u8, x27, x33);
  let mut x39: u64 = 0;
  let mut x40: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x39, &mut x40, x38, x29, x35);
  let mut x41: u64 = 0;
  let mut x42: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x41, &mut x42, x40, x31, x36);
  let x43: u64 = ((x42 as u64) + (x32 as u64));
  let mut x44: u64 = 0;
  let mut x45: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x44, &mut x45, 0x0_u8, x39, (0x1_u8 as u64));
  let mut x46: u64 = 0;
  let mut x47: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x46, &mut x47, x45, x41, 0xffffffffffffffe4_u64);
  let mut x48: u64 = 0;
  let mut x49: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x48, &mut x49, x47, x43, (0x0_u8 as u64));
  let mut x50: u64 = 0;
  fp128_cmovznz_u64(&mut x50, x49, x44, x39);
  let mut x51: u64 = 0;
  fp128_cmovznz_u64(&mut x51, x49, x46, x41);
  out1[0] = x50;
  out1[1] = x51;
}

/// The function fp128_opp negates a field element in the Montgomery domain.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
/// Postconditions:
///   eval (from_montgomery out1) mod m = -eval (from_montgomery arg1) mod m
///   0 ≤ eval out1 < m
///
#[inline(always)]
pub fn fp128_opp(out1: &mut Fp128MontgomeryDomainFieldElement, arg1: &Fp128MontgomeryDomainFieldElement) {
  let mut x1: u64 = 0;
  let mut x2: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x1, &mut x2, 0x0_u8, (0x0_u8 as u64), (arg1[0]));
  let mut x3: u64 = 0;
  let mut x4: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x3, &mut x4, x2, (0x0_u8 as u64), (arg1[1]));
  let mut x5: u64 = 0;
  fp128_cmovznz_u64(&mut x5, x4, (0x0_u8 as u64), 0xffffffffffffffff_u64);
  let mut x6: u64 = 0;
  let mut x7: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x6, &mut x7, 0x0_u8, x1, (((x5 & (0x1_u8 as u64)) as Fp128U1) as u64));
  let mut x8: u64 = 0;
  let mut x9: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x8, &mut x9, x7, x3, (x5 & 0xffffffffffffffe4_u64));
  out1[0] = x6;
  out1[1] = x8;
}

/// The function fp128_from_montgomery translates a field element out of the Montgomery domain.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
/// Postconditions:
///   eval out1 mod m = (eval arg1 * ((2^64)⁻¹ mod m)^2) mod m
///   0 ≤ eval out1 < m
///
#[inline(always)]
pub fn fp128_from_montgomery(out1: &mut Fp128NonMontgomeryDomainFieldElement, arg1: &Fp128MontgomeryDomainFieldElement) {
  let x1: u64 = (arg1[0]);
  let mut x2: u64 = 0;
  let mut x3: u64 = 0;
  fp128_mulx_u64(&mut x2, &mut x3, x1, 0xffffffffffffffff_u64);
  let mut x4: u64 = 0;
  let mut x5: u64 = 0;
  fp128_mulx_u64(&mut x4, &mut x5, x2, 0xffffffffffffffe4_u64);
  let mut x6: u64 = 0;
  let mut x7: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x6, &mut x7, 0x0_u8, x1, x2);
  let mut x8: u64 = 0;
  let mut x9: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x8, &mut x9, x7, (0x0_u8 as u64), x4);
  let mut x10: u64 = 0;
  let mut x11: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x10, &mut x11, 0x0_u8, x8, (arg1[1]));
  let mut x12: u64 = 0;
  let mut x13: u64 = 0;
  fp128_mulx_u64(&mut x12, &mut x13, x10, 0xffffffffffffffff_u64);
  let mut x14: u64 = 0;
  let mut x15: u64 = 0;
  fp128_mulx_u64(&mut x14, &mut x15, x12, 0xffffffffffffffe4_u64);
  let mut x16: u64 = 0;
  let mut x17: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x16, &mut x17, 0x0_u8, x10, x12);
  let mut x18: u64 = 0;
  let mut x19: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x18, &mut x19, x17, ((x11 as u64) + ((x9 as u64) + x5)), x14);
  let x20: u64 = ((x19 as u64) + x15);
  let mut x21: u64 = 0;
  let mut x22: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x21, &mut x22, 0x0_u8, x18, (0x1_u8 as u64));
  let mut x23: u64 = 0;
  let mut x24: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x23, &mut x24, x22, x20, 0xffffffffffffffe4_u64);
  let mut x25: u64 = 0;
  let mut x26: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x25, &mut x26, x24, (0x0_u8 as u64), (0x0_u8 as u64));
  let mut x27: u64 = 0;
  fp128_cmovznz_u64(&mut x27, x26, x21, x18);
  let mut x28: u64 = 0;
  fp128_cmovznz_u64(&mut x28, x26, x23, x20);
  out1[0] = x27;
  out1[1] = x28;
}

/// The function fp128_to_montgomery translates a field element into the Montgomery domain.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
/// Postconditions:
///   eval (from_montgomery out1) mod m = eval arg1 mod m
///   0 ≤ eval out1 < m
///
#[inline(always)]
pub fn fp128_to_montgomery(out1: &mut Fp128MontgomeryDomainFieldElement, arg1: &Fp128NonMontgomeryDomainFieldElement) {
  let x1: u64 = (arg1[1]);
  let x2: u64 = (arg1[0]);
  let mut x3: u64 = 0;
  let mut x4: u64 = 0;
  fp128_mulx_u64(&mut x3, &mut x4, x2, 0x5587_u64);
  let mut x5: u64 = 0;
  let mut x6: u64 = 0;
  fp128_mulx_u64(&mut x5, &mut x6, x2, 0xfffffffffffffcf1_u64);
  let mut x7: u64 = 0;
  let mut x8: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x7, &mut x8, 0x0_u8, x6, x3);
  let mut x9: u64 = 0;
  let mut x10: u64 = 0;
  fp128_mulx_u64(&mut x9, &mut x10, x5, 0xffffffffffffffff_u64);
  let mut x11: u64 = 0;
  let mut x12: u64 = 0;
  fp128_mulx_u64(&mut x11, &mut x12, x9, 0xffffffffffffffe4_u64);
  let mut x13: u64 = 0;
  let mut x14: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x13, &mut x14, 0x0_u8, x5, x9);
  let mut x15: u64 = 0;
  let mut x16: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x15, &mut x16, x14, x7, x11);
  let mut x17: u64 = 0;
  let mut x18: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x17, &mut x18, x16, ((x8 as u64) + x4), x12);
  let mut x19: u64 = 0;
  let mut x20: u64 = 0;
  fp128_mulx_u64(&mut x19, &mut x20, x1, 0x5587_u64);
  let mut x21: u64 = 0;
  let mut x22: u64 = 0;
  fp128_mulx_u64(&mut x21, &mut x22, x1, 0xfffffffffffffcf1_u64);
  let mut x23: u64 = 0;
  let mut x24: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x23, &mut x24, 0x0_u8, x22, x19);
  let mut x25: u64 = 0;
  let mut x26: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x25, &mut x26, 0x0_u8, x15, x21);
  let mut x27: u64 = 0;
  let mut x28: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x27, &mut x28, x26, x17, x23);
  let mut x29: u64 = 0;
  let mut x30: u64 = 0;
  fp128_mulx_u64(&mut x29, &mut x30, x25, 0xffffffffffffffff_u64);
  let mut x31: u64 = 0;
  let mut x32: u64 = 0;
  fp128_mulx_u64(&mut x31, &mut x32, x29, 0xffffffffffffffe4_u64);
  let mut x33: u64 = 0;
  let mut x34: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x33, &mut x34, 0x0_u8, x25, x29);
  let mut x35: u64 = 0;
  let mut x36: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x35, &mut x36, x34, x27, x31);
  let mut x37: u64 = 0;
  let mut x38: Fp128U1 = 0;
  fp128_addcarryx_u64(&mut x37, &mut x38, x36, (((x28 as u64) + (x18 as u64)) + ((x24 as u64) + x20)), x32);
  let mut x39: u64 = 0;
  let mut x40: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x39, &mut x40, 0x0_u8, x35, (0x1_u8 as u64));
  let mut x41: u64 = 0;
  let mut x42: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x41, &mut x42, x40, x37, 0xffffffffffffffe4_u64);
  let mut x43: u64 = 0;
  let mut x44: Fp128U1 = 0;
  fp128_subborrowx_u64(&mut x43, &mut x44, x42, (x38 as u64), (0x0_u8 as u64));
  let mut x45: u64 = 0;
  fp128_cmovznz_u64(&mut x45, x44, x39, x35);
  let mut x46: u64 = 0;
  fp128_cmovznz_u64(&mut x46, x44, x41, x37);
  out1[0] = x45;
  out1[1] = x46;
}

/// The function fp128_to_bytes serializes a field element NOT in the Montgomery domain to bytes in little-endian order.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
/// Postconditions:
///   out1 = map (λ x, ⌊((eval arg1 mod m) mod 2^(8 * (x + 1))) / 2^(8 * x)⌋) [0..15]
///
/// Input Bounds:
///   arg1: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
/// Output Bounds:
///   out1: [[0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff]]
#[inline(always)]
pub fn fp128_to_bytes(out1: &mut [u8; 16], arg1: &[u64; 2]) {
  let x1: u64 = (arg1[1]);
  let x2: u64 = (arg1[0]);
  let x3: u8 = ((x2 & (0xff_u8 as u64)) as u8);
  let x4: u64 = (x2 >> 8);
  let x5: u8 = ((x4 & (0xff_u8 as u64)) as u8);
  let x6: u64 = (x4 >> 8);
  let x7: u8 = ((x6 & (0xff_u8 as u64)) as u8);
  let x8: u64 = (x6 >> 8);
  let x9: u8 = ((x8 & (0xff_u8 as u64)) as u8);
  let x10: u64 = (x8 >> 8);
  let x11: u8 = ((x10 & (0xff_u8 as u64)) as u8);
  let x12: u64 = (x10 >> 8);
  let x13: u8 = ((x12 & (0xff_u8 as u64)) as u8);
  let x14: u64 = (x12 >> 8);
  let x15: u8 = ((x14 & (0xff_u8 as u64)) as u8);
  let x16: u8 = ((x14 >> 8) as u8);
  let x17: u8 = ((x1 & (0xff_u8 as u64)) as u8);
  let x18: u64 = (x1 >> 8);
  let x19: u8 = ((x18 & (0xff_u8 as u64)) as u8);
  let x20: u64 = (x18 >> 8);
  let x21: u8 = ((x20 & (0xff_u8 as u64)) as u8);
  let x22: u64 = (x20 >> 8);
  let x23: u8 = ((x22 & (0xff_u8 as u64)) as u8);
  let x24: u64 = (x22 >> 8);
  let x25: u8 = ((x24 & (0xff_u8 as u64)) as u8);
  let x26: u64 = (x24 >> 8);
  let x27: u8 = ((x26 & (0xff_u8 as u64)) as u8);
  let x28: u64 = (x26 >> 8);
  let x29: u8 = ((x28 & (0xff_u8 as u64)) as u8);
  let x30: u8 = ((x28 >> 8) as u8);
  out1[0] = x3;
  out1[1] = x5;
  out1[2] = x7;
  out1[3] = x9;
  out1[4] = x11;
  out1[5] = x13;
  out1[6] = x15;
  out1[7] = x16;
  out1[8] = x17;
  out1[9] = x19;
  out1[10] = x21;
  out1[11] = x23;
  out1[12] = x25;
  out1[13] = x27;
  out1[14] = x29;
  out1[15] = x30;
}

/// The function fp128_from_bytes deserializes a field element NOT in the Montgomery domain from bytes in little-endian order.
///
/// Preconditions:
///   0 ≤ bytes_eval arg1 < m
/// Postconditions:
///   eval out1 mod m = bytes_eval arg1 mod m
///   0 ≤ eval out1 < m
///
/// Input Bounds:
///   arg1: [[0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff]]
/// Output Bounds:
///   out1: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
#[inline(always)]
pub fn fp128_from_bytes(out1: &mut [u64; 2], arg1: &[u8; 16]) {
  let x1: u64 = (((arg1[15]) as u64) << 56);
  let x2: u64 = (((arg1[14]) as u64) << 48);
  let x3: u64 = (((arg1[13]) as u64) << 40);
  let x4: u64 = (((arg1[12]) as u64) << 32);
  let x5: u64 = (((arg1[11]) as u64) << 24);
  let x6: u64 = (((arg1[10]) as u64) << 16);
  let x7: u64 = (((arg1[9]) as u64) << 8);
  let x8: u8 = (arg1[8]);
  let x9: u64 = (((arg1[7]) as u64) << 56);
  let x10: u64 = (((arg1[6]) as u64) << 48);
  let x11: u64 = (((arg1[5]) as u64) << 40);
  let x12: u64 = (((arg1[4]) as u64) << 32);
  let x13: u64 = (((arg1[3]) as u64) << 24);
  let x14: u64 = (((arg1[2]) as u64) << 16);
  let x15: u64 = (((arg1[1]) as u64) << 8);
  let x16: u8 = (arg1[0]);
  let x17: u64 = (x15 + (x16 as u64));
  let x18: u64 = (x14 + x17);
  let x19: u64 = (x13 + x18);
  let x20: u64 = (x12 + x19);
  let x21: u64 = (x11 + x20);
  let x22: u64 = (x10 + x21);
  let x23: u64 = (x9 + x22);
  let x24: u64 = (x7 + (x8 as u64));
  let x25: u64 = (x6 + x24);
  let x26: u64 = (x5 + x25);
  let x27: u64 = (x4 + x26);
  let x28: u64 = (x3 + x27);
  let x29: u64 = (x2 + x28);
  let x30: u64 = (x1 + x29);
  out1[0] = x23;
  out1[1] = x30;
}
