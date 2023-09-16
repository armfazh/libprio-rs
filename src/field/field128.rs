// Copyright (c) 2023 ISRG
// SPDX-License-Identifier: MPL-2.0

//! Finite field arithmetic for `GF((2^62-2^3+1)*2^66+1)`.

use super::{
    fiat_crypto_fp128::{
        fp128_add, fp128_from_bytes, fp128_from_montgomery, fp128_mul, fp128_opp, fp128_sub,
        fp128_to_bytes, fp128_to_montgomery, Fp128MontgomeryDomainFieldElement,
        Fp128NonMontgomeryDomainFieldElement,
    },
    FftFriendlyFieldElement, FieldElement, FieldElementVisitor, FieldElementWithInteger,
    FieldError,
};
use crate::codec::{CodecError, Decode, Encode};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    convert::{TryFrom, TryInto},
    fmt::{Debug, Display, Formatter},
    hash::{Hash, Hasher},
    io::{Cursor, Read},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

impl From<Fp128NonMontgomeryDomainFieldElement> for u128 {
    fn from(value: Fp128NonMontgomeryDomainFieldElement) -> Self {
        ((value.0[1] as u128) << 64) | (value.0[0] as u128)
    }
}

impl From<u128> for Fp128NonMontgomeryDomainFieldElement {
    fn from(val: u128) -> Self {
        Fp128NonMontgomeryDomainFieldElement([val as u64, (val >> 64) as u64])
    }
}

impl From<Fp128MontgomeryDomainFieldElement> for Fp128NonMontgomeryDomainFieldElement {
    fn from(val: Fp128MontgomeryDomainFieldElement) -> Self {
        let mut out = Fp128NonMontgomeryDomainFieldElement(Default::default());
        fp128_from_montgomery(&mut out, &val);
        out
    }
}

impl From<Fp128NonMontgomeryDomainFieldElement> for Fp128MontgomeryDomainFieldElement {
    fn from(val: Fp128NonMontgomeryDomainFieldElement) -> Self {
        let mut out = Fp128MontgomeryDomainFieldElement(Default::default());
        fp128_to_montgomery(&mut out, &val);
        out
    }
}

impl PartialEq for Fp128MontgomeryDomainFieldElement {
    fn eq(&self, rhs: &Self) -> bool {
        self.0[0] == rhs.0[0] && self.0[1] == rhs.0[1]
    }
}

impl Eq for Fp128MontgomeryDomainFieldElement {}

impl Hash for Fp128MontgomeryDomainFieldElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]

/// Field128 represents an element of the prime field GF((2^62-2^3+1)*2^66+1).
///
/// Internally, elements use Montgomery representation and its
/// implementation is provided by fiat-crypto tool that produces
/// formally-verified prime field arithmetic.
pub struct Field128(Fp128MontgomeryDomainFieldElement);

macro_rules! makeFp {
    ($u0:expr,$u1:expr) => {
        Field128(Fp128MontgomeryDomainFieldElement([$u0, $u1]))
    };
}

impl Field128 {
    // fn try_from_bytes(bytes: &[u8], mask: u128) -> Result<Self, FieldError> {
    //     if Self::ENCODED_SIZE > bytes.len() {
    //         return Err(FieldError::ShortRead);
    //     }

    //     let mut int = 0;
    //     for i in 0..Self::ENCODED_SIZE {
    //         int |= (bytes[i] as u128) << (i << 3);
    //     }

    //     int &= mask;

    //     if int >= FP128_PARAMS.p {
    //         return Err(FieldError::ModulusOverflow);
    //     }
    //     // FieldParameters::montgomery() will return a value that has been fully reduced
    //     // mod p, satisfying the invariant on Self.
    //     Ok(Self(FP128_PARAMS.montgomery(int)))
    // }
}

impl Default for Field128 {
    #[inline]
    fn default() -> Self {
        makeFp!(u64::default(), u64::default())
    }
}

impl ConstantTimeEq for Field128 {
    #[inline]
    fn ct_eq(&self, rhs: &Self) -> Choice {
        u64::ct_eq(&self.0[0], &rhs.0[0]) & u64::ct_eq(&self.0[1], &rhs.0[1])
    }
}

impl ConditionallySelectable for Field128 {
    #[inline]
    fn conditional_select(a: &Self, b: &Self, c: Choice) -> Self {
        makeFp!(
            u64::conditional_select(&a.0[0], &b.0[0], c),
            u64::conditional_select(&a.0[1], &b.0[1], c)
        )
    }
}

impl Add for Field128 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = Field128::default();
        fp128_add(&mut out.0, &self.0, &rhs.0);
        out
    }
}

impl Add for &Field128 {
    type Output = Field128;
    fn add(self, rhs: Self) -> Self::Output {
        *self + *rhs
    }
}

impl AddAssign for Field128 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Field128 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = Field128::default();
        fp128_sub(&mut out.0, &self.0, &rhs.0);
        out
    }
}

impl Sub for &Field128 {
    type Output = Field128;
    fn sub(self, rhs: Self) -> Self::Output {
        *self - *rhs
    }
}

impl SubAssign for Field128 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for Field128 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut out = Field128::default();
        fp128_mul(&mut out.0, &self.0, &rhs.0);
        out
    }
}

impl Mul for &Field128 {
    type Output = Field128;
    fn mul(self, rhs: Self) -> Self::Output {
        *self * *rhs
    }
}

impl MulAssign for Field128 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Field128 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.mul(rhs.inv())
    }
}

impl Div for &Field128 {
    type Output = Field128;
    fn div(self, rhs: Self) -> Self::Output {
        *self / *rhs
    }
}

impl DivAssign for Field128 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Neg for Field128 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut out = Field128::default();
        fp128_opp(&mut out.0, &self.0);
        out
    }
}

impl Neg for &Field128 {
    type Output = Field128;
    fn neg(self) -> Self::Output {
        -(*self)
    }
}

impl From<u128> for Field128 {
    fn from(_x: u128) -> Self {
        // must check that u128 is full reduced, or decide to reduce?
        // u128 => bytes => Field128
        // u128::try_from(x).unwrap()
        unimplemented!()
    }
}

impl From<Field128> for u128 {
    fn from(x: Field128) -> Self {
        Into::<u128>::into(Into::<Fp128NonMontgomeryDomainFieldElement>::into(x.0))
    }
}

impl<'a> TryFrom<&'a [u8]> for Field128 {
    type Error = FieldError;

    fn try_from(bytes: &[u8]) -> Result<Self, FieldError> {
        let mut in_bytes = [0u8; FIELD128_ENCODED_SIZE];
        in_bytes.copy_from_slice(bytes);
        let mut value = [0u64; 2];
        fp128_from_bytes(&mut value, &in_bytes);
        let value_u128: u128 = ((value[1] as u128) << 64) | (value[0] as u128);
        // (todo) must check value is in range.
        if value_u128 > 0 {
            return Err(FieldError::IntegerTryFrom);
        }
        Ok(Self(Into::<Fp128MontgomeryDomainFieldElement>::into(
            Into::<Fp128NonMontgomeryDomainFieldElement>::into(value_u128),
        )))
    }
}

impl From<Field128> for [u8; FIELD128_ENCODED_SIZE] {
    fn from(elem: Field128) -> Self {
        let mut slice = Self::default();
        fp128_to_bytes(
            &mut slice,
            &Into::<Fp128NonMontgomeryDomainFieldElement>::into(elem.0).0,
        );
        slice
    }
}

impl From<Field128> for Vec<u8> {
    fn from(elem: Field128) -> Self {
        <[u8; FIELD128_ENCODED_SIZE]>::from(elem).to_vec()
    }
}

impl Display for Field128 {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", u128::from(*self))
    }
}

impl Debug for Field128 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", u128::from(*self))
    }
}

impl Serialize for Field128 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let bytes: [u8; FIELD128_ENCODED_SIZE] = (*self).into();
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for Field128 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_bytes(FieldElementVisitor {
            phantom: PhantomData,
        })
    }
}

impl Encode for Field128 {
    fn encode(&self, bytes: &mut Vec<u8>) {
        let slice = <[u8; FIELD128_ENCODED_SIZE]>::from(*self);
        bytes.extend_from_slice(&slice);
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(FIELD128_ENCODED_SIZE)
    }
}

impl Decode for Field128 {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let mut value = [0u8; FIELD128_ENCODED_SIZE];
        bytes.read_exact(&mut value)?;
        Field128::try_from(value.as_slice()).map_err(|e| {
            CodecError::Other(Box::new(e) as Box<dyn std::error::Error + 'static + Send + Sync>)
        })
    }
}

impl FieldElement for Field128 {
    const ENCODED_SIZE: usize = FIELD128_ENCODED_SIZE;

    #[inline]
    fn inv(&self) -> Self {
        self.pow(FIELD128_PRIME - 2)
    }

    fn try_from_random(_bytes: &[u8]) -> Result<Self, FieldError> {
        // todo: decide how to acccept input from a random vector.
        // Field128::try_from(bytes, FP128_PARAMS.bit_mask)
        unimplemented!()
    }

    #[inline]
    fn zero() -> Self {
        Self::default()
    }

    #[inline]
    fn one() -> Self {
        // Since FIELD128_UROOTS[i] is the root-of-unity to the i-th power,
        // so FIELD128_UROOTS[0] = 1.
        FIELD128_UROOTS[0]
    }
}

impl FieldElementWithInteger for Field128 {
    type Integer = u128;
    type IntegerTryFromError = <Self::Integer as TryFrom<usize>>::Error;
    type TryIntoU64Error = <Self::Integer as TryInto<u64>>::Error;

    // pow is a non-constant-time operation with respect to the bits of the exponent.
    fn pow(&self, exp: Self::Integer) -> Self {
        let mut t = Self::one();
        for i in (0..u128::BITS - exp.leading_zeros()).rev() {
            t *= t;
            if (exp >> i) & 1 != 0 {
                t *= *self;
            }
        }
        t
    }

    fn modulus() -> Self::Integer {
        FIELD128_PRIME
    }
}

impl FftFriendlyFieldElement for Field128 {
    fn generator() -> Self {
        FIELD128_PRIMITIVE_UROOT
    }

    fn generator_order() -> Self::Integer {
        1 << FIELD128_NUM_UROOTS
    }

    fn root(l: usize) -> Option<Self> {
        if l < FIELD128_UROOTS.len() {
            Some(FIELD128_UROOTS[l])
        } else {
            None
        }
    }
}

/// The prime modulus `p=(2^62-2^3+1)*2^66+1`.
const FIELD128_PRIME: u128 = 340282366920938462946865773367900766209;
/// Size in bytes used to store a Field128 element.
const FIELD128_ENCODED_SIZE: usize = 16;
/// The `2^num_roots`-th -principal root of unity. This element is used to generate the
/// elements of `roots`.
///
/// In sage this is calculated as:
///   PRIMITIVE_UROOT = GF(p).primitive_element()^(2^62-2^3+1)
///                   = 7^(2^62-2^3+1)
///                   = 145091266659756586618791329697897684742
/// Then, converted to Montgomery domain with R=2^128.
///   FIELD128_PRIMITIVE_UROOT = PRIMITIVE_UROOT * 2^128
///                            = 0x50f8f7f554db309cf0111fb98c6b9875
// Field128(Fp128MontgomeryDomainFieldElement([
static FIELD128_PRIMITIVE_UROOT: Field128 = makeFp!(0xf0111fb98c6b9875, 0x50f8f7f554db309c);
/// The number of principal roots of unity in `roots`.
const FIELD128_NUM_UROOTS: usize = 66;
/// `FIELD128_UROOTS[l]` is the `2^l`-th principal root of unity, i.e., `roots[l]` has order `2^l` in the
/// multiplicative group. `roots[0]` is equal to one by definition.
///
/// In sage this is calculated as:
///   PRIMITIVE_UROOT = GF(p).primitive_element()^(2^62-2^3+1)
///   toMont = lambda x: x*2**128
///   toHex = lambda x: list(map(hex, ZZ(x).digits(2**64)))
///   FIELD128_UROOTS = [ toHex(toMont(b**i)) for i in range(0,21) ]
static FIELD128_UROOTS: [Field128; 20 + 1] = [
    makeFp!(0xffffffffffffffff, 0x1b),
    makeFp!(0xf0111fb98c6b9875, 0x50f8f7f554db309c),
    makeFp!(0x336824df50a3e9de, 0x1f70898af1701972),
    makeFp!(0x52084516b37d72db, 0x8bfac52aac2c36e9),
    makeFp!(0x8b45f6e90b16542d, 0xd9b5db39af523ffb),
    makeFp!(0xec875e5c00353b61, 0x1bd0dcb83d7ea68a),
    makeFp!(0x4444393ff6c7e30c, 0xa71f88583de4e579),
    makeFp!(0x259ae62476ae0522, 0xc38d70695f91561e),
    makeFp!(0x3c5dd0cda73814c2, 0x1a29ef0cd721372e),
    makeFp!(0xf5df5ca55e4b2158, 0xa8f54fcb7822853f),
    makeFp!(0x2b5a0f0e88eda7fa, 0x327e63c2205a06ae),
    makeFp!(0x5db505eaaab5261d, 0xdb7a4c65816d8488),
    makeFp!(0x64ed61974e585c72, 0x7dab815089a2a138),
    makeFp!(0x6cf557b0cb3a2f3b, 0x3cff2f56bf6877d7),
    makeFp!(0x2b809b5aeb580d92, 0xba46c3968632b291),
    makeFp!(0x20fd8b7d3df3a711, 0x2bb109cbdaafa592),
    makeFp!(0x9e4fc0f4006111e3, 0x54879fe858345e92),
    makeFp!(0xdaadd17a3e528054, 0x5a99e1f583619898),
    makeFp!(0x15643d3043bfe8d6, 0x88a9b82ef788b332),
    makeFp!(0xfd5ce8b146115bc7, 0x2bed551a665599c8),
    makeFp!(0x996ef60497805c22, 0x853e4df8bbb45538),
];
