// Copyright (c) 2023 ISRG
// SPDX-License-Identifier: MPL-2.0

//! Finite field arithmetic for `GF((2^62-2^3+1)*2^66+1)`.

use crate::codec::{CodecError, Decode, Encode};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    cmp::min,
    convert::{TryFrom, TryInto},
    fmt::{Debug, Display, Formatter},
    hash::{Hash, Hasher},
    io::{Cursor, Read},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::fp::FieldParameters;

use super::{
    fiat_crypto_fp128::{
        fp128_add, fp128_from_bytes, fp128_from_montgomery, fp128_mul, fp128_opp, fp128_set_one,
        fp128_sub, fp128_to_bytes, fp128_to_montgomery, Fp128MontgomeryDomainFieldElement,
        Fp128NonMontgomeryDomainFieldElement,
    },
    FftFriendlyFieldElement, FieldElement, FieldElementVisitor, FieldElementWithInteger,
    FieldError,
};

// (todo) replace these constants with Montgomery representation to avoid multiplications.
const FP128_PARAMS: FieldParameters = FieldParameters {
    p: 340282366920938462946865773367900766209, // 128-bit prime
    mu: 18446744073709551615,
    r2: 403909908237944342183153,
    g: 107630958476043550189608038630704257141,
    num_roots: 66,
    bit_mask: 340282366920938463463374607431768211455,
    roots: [
        516508834063867445247,
        340282366920938462430356939304033320962,
        129526470195413442198896969089616959958,
        169031622068548287099117778531474117974,
        81612939378432101163303892927894236156,
        122401220764524715189382260548353967708,
        199453575871863981432000940507837456190,
        272368408887745135168960576051472383806,
        24863773656265022616993900367764287617,
        257882853788779266319541142124730662203,
        323732363244658673145040701829006542956,
        57532865270871759635014308631881743007,
        149571414409418047452773959687184934208,
        177018931070866797456844925926211239962,
        268896136799800963964749917185333891349,
        244556960591856046954834420512544511831,
        118945432085812380213390062516065622346,
        202007153998709986841225284843501908420,
        332677126194796691532164818746739771387,
        258279638927684931537542082169183965856,
        148221243758794364405224645520862378432,
    ],
};

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

/// Field128 represents a field element in the prime field GF((2^62-2^3+1)*2^66+1).
///
/// Internally, elements use Montgomery representation and its
/// implementation is provided by fiat-crypto, a formally-verified
/// tool that produces prime field arithmetic.
pub struct Field128(Fp128MontgomeryDomainFieldElement);

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
    fn default() -> Self {
        Self(Fp128MontgomeryDomainFieldElement(Default::default()))
    }
}

impl ConstantTimeEq for Field128 {
    fn ct_eq(&self, rhs: &Self) -> Choice {
        u64::ct_eq(&self.0[0], &rhs.0[0]) & u64::ct_eq(&self.0[1], &rhs.0[1])
    }
}

impl ConditionallySelectable for Field128 {
    fn conditional_select(a: &Self, b: &Self, choice: subtle::Choice) -> Self {
        Self(Fp128MontgomeryDomainFieldElement([
            u64::conditional_select(&a.0[0], &b.0[0], choice),
            u64::conditional_select(&a.0[1], &b.0[1], choice),
        ]))
    }
}

impl Add for Field128 {
    type Output = Field128;
    fn add(self, rhs: Self) -> Self {
        let mut out = Field128::default();
        fp128_add(&mut out.0, &self.0, &rhs.0);
        out
    }
}

impl Add for &Field128 {
    type Output = Field128;
    fn add(self, rhs: Self) -> Field128 {
        *self + *rhs
    }
}

impl AddAssign for Field128 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Field128 {
    type Output = Field128;
    fn sub(self, rhs: Self) -> Self {
        let mut out = Field128::default();
        fp128_sub(&mut out.0, &self.0, &rhs.0);
        out
    }
}

impl Sub for &Field128 {
    type Output = Field128;
    fn sub(self, rhs: Self) -> Field128 {
        *self - *rhs
    }
}

impl SubAssign for Field128 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for Field128 {
    type Output = Field128;
    fn mul(self, rhs: Self) -> Self {
        let mut out = Field128::default();
        fp128_mul(&mut out.0, &self.0, &rhs.0);
        out
    }
}

impl Mul for &Field128 {
    type Output = Field128;
    fn mul(self, rhs: Self) -> Field128 {
        *self * *rhs
    }
}

impl MulAssign for Field128 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Field128 {
    type Output = Field128;
    fn div(self, rhs: Self) -> Self {
        self.mul(rhs.inv())
    }
}

impl Div for &Field128 {
    type Output = Field128;
    fn div(self, rhs: Self) -> Field128 {
        *self / *rhs
    }
}

impl DivAssign for Field128 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Neg for Field128 {
    type Output = Field128;
    fn neg(self) -> Self {
        let mut out = Field128::default();
        fp128_opp(&mut out.0, &self.0);
        out
    }
}

impl Neg for &Field128 {
    type Output = Field128;
    fn neg(self) -> Field128 {
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
        let mut in_bytes = [0u8; Field128::ENCODED_SIZE];
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

impl From<Field128> for [u8; Field128::ENCODED_SIZE] {
    fn from(elem: Field128) -> Self {
        let mut slice = [0; Field128::ENCODED_SIZE];
        fp128_to_bytes(
            &mut slice,
            &Into::<Fp128NonMontgomeryDomainFieldElement>::into(elem.0).0,
        );
        slice
    }
}

impl From<Field128> for Vec<u8> {
    fn from(elem: Field128) -> Self {
        <[u8; Field128::ENCODED_SIZE]>::from(elem).to_vec()
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
        let bytes: [u8; Field128::ENCODED_SIZE] = (*self).into();
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for Field128 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Field128, D::Error> {
        deserializer.deserialize_bytes(FieldElementVisitor {
            phantom: PhantomData,
        })
    }
}

impl Encode for Field128 {
    fn encode(&self, bytes: &mut Vec<u8>) {
        let slice = <[u8; Field128::ENCODED_SIZE]>::from(*self);
        bytes.extend_from_slice(&slice);
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(Self::ENCODED_SIZE)
    }
}

impl Decode for Field128 {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let mut value = [0u8; Field128::ENCODED_SIZE];
        bytes.read_exact(&mut value)?;
        Field128::try_from(value.as_slice()).map_err(|e| {
            CodecError::Other(Box::new(e) as Box<dyn std::error::Error + 'static + Send + Sync>)
        })
    }
}

impl FieldElement for Field128 {
    const ENCODED_SIZE: usize = 16;

    #[inline]
    fn inv(&self) -> Self {
        self.pow(FP128_PARAMS.p - 2)
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
        let mut uno = Self::default();
        fp128_set_one(&mut uno.0);
        uno
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
        FP128_PARAMS.p
    }
}

impl FftFriendlyFieldElement for Field128 {
    fn generator() -> Self {
        // todo: need to avoid multiplication during conversion
        FP128_PARAMS.g.into()
    }

    fn generator_order() -> Self::Integer {
        1 << (Self::Integer::try_from(FP128_PARAMS.num_roots).unwrap())
    }

    fn root(l: usize) -> Option<Self> {
        // todo: need to avoid multiplication during conversion
        if l < min(FP128_PARAMS.roots.len(), FP128_PARAMS.num_roots + 1) {
            Some(FP128_PARAMS.roots[l].into())
        } else {
            None
        }
    }
}
