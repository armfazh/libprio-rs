// SPDX-License-Identifier: MPL-2.0

//! Prime field arithmetic for any Galois field GF(p) for which `p < 2^W`,
//! where `W ≤ 64` is a specified word size.

use fixed::traits::FixedBits;
use num_traits::{
    ops::overflowing::{OverflowingAdd, OverflowingSub},
    AsPrimitive, ConstOne, ConstZero, Unsigned, WrappingAdd, WrappingMul, WrappingSub,
};

use crate::fp::MAX_ROOTS;

pub(crate) trait Word:
    Unsigned
    + FixedBits
    + OverflowingAdd
    + OverflowingSub
    + WrappingAdd
    + WrappingSub
    + WrappingMul
    + ConstZero
    + ConstOne
    + AsPrimitive<Self::Wide>
where
    bool: AsPrimitive<Self> + AsPrimitive<Self::Wide>,
{
    type Wide: Unsigned + FixedBits + AsPrimitive<Self>;

    fn leading_zeros(self) -> u32;
}

/// `FieldParametersSingleWord` has parameters of a prime field GF(p)
/// for which the prime modulus `p` fits in one word of `W ≤ 64` bits.
/// It requires a multiplication operation on unsigned integers.
///
/// See also [`FieldParameters`](crate::fp::FieldParameters).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FieldParametersSingleWord<W>
where
    W: Word,
    bool: AsPrimitive<W> + AsPrimitive<W::Wide>,
{
    /// The prime modulus `p`.
    pub p: W,
    /// `mu = -p^(-1) mod 2^W`.
    pub mu: W,
    /// `r2 = (2^W)^2 mod p`.
    pub r2: W,
    /// The `2^num_roots`-th -principal root of unity. This element
    /// is used to generate the elements of `roots`.
    pub g: W,
    /// The number of principal roots of unity in `roots`.
    pub num_roots: usize,
    /// Equal to `2^b - 1`, where `b` is the length of `p` in bits.
    pub bit_mask: W,
    /// `roots[l]` is the `2^l`-th principal root of unity, i.e.,
    /// `roots[l]` has order `2^l` in the multiplicative group.
    /// `roots[0]` is equal to one by definition.
    pub roots: [W; MAX_ROOTS + 1],
}

impl<W> FieldParametersSingleWord<W>
where
    W: Word,
    bool: AsPrimitive<W> + AsPrimitive<W::Wide>,
{
    /// Addition. The result will be in [0, p), so long as both x
    /// and y are as well.
    #[inline(always)]
    pub(crate) fn add(&self, x: W, y: W) -> W {
        //   0,x
        // + 0,y
        // =====
        //   c,z
        let (z, carry) = x.overflowing_add(&y);

        //     c, z
        // -   0, p
        // ========
        // b1,s1,s0
        let (s0, b0) = z.overflowing_sub(&self.p);
        let (_s1, b1) = AsPrimitive::<W>::as_(carry).overflowing_sub(&b0.as_());
        // if b1 == 1: return z
        // else:       return s0
        let mask = W::ZERO.wrapping_sub(&b1.as_());
        (z & mask) | (s0 & !mask)
    }

    /// Subtraction. The result will be in [0, p), so long as both x and y are as well.
    #[inline(always)]
    pub(crate) fn sub(&self, x: W, y: W) -> W {
        //        x
        // -      y
        // ========
        //    b0,z0
        let (z0, b0) = x.overflowing_sub(&y);
        let m = W::ZERO.wrapping_sub(&b0.as_());
        //      z0
        // +     p
        // ========
        //   s1,s0
        z0.wrapping_add(&(m & self.p))
        // z0 + (m & self.p)
        // if b1 == 1: return s0
        // else:       return z0
    }

    #[inline(always)]
    fn modp(&self, x: W) -> W {
        self.sub(x, self.p)
    }

    /// Multiplication of field elements in the Montgomery domain.
    /// This uses the Montgomery's [REDC algorithm][montgomery].
    /// The result will be in [0, p).
    ///
    /// [montgomery]: https://www.ams.org/journals/mcom/1985-44-170/S0025-5718-1985-0777282-X/S0025-5718-1985-0777282-X.pdf
    #[inline(always)]
    pub(crate) fn mul(&self, x: W, y: W) -> W {
        let hi_lo = |v: W::Wide| ((v >> W::BITS).as_(), v.as_());

        // Integer multiplication
        // z = x * y

        //     x
        // *   y
        // =====
        // z1,z0
        let (z1, z0) = hi_lo(x.as_() * y.as_());

        // Montgomery Reduction
        // z = z + p * mu*(z mod 2^W), where mu = (-p)^(-1) mod 2^W.
        //   = z + p * w, where w = mu*z0
        let w = self.mu.wrapping_mul(&z0);
        let (r1, r0) = hi_lo(self.p.as_() * w.as_());

        //    z1,z0
        // +  r1,r0
        //    =====
        // cc, z, 0
        let (_zero, carry) = z0.overflowing_add(&r0);
        let (cc, z) = hi_lo(z1.as_() + r1.as_() + carry.as_());

        // Final subtraction
        // If z >= p, then z = z - p

        //    cc, z
        // -   0, p
        // ========
        // b1,s1,s0
        let (s0, b0) = z.overflowing_sub(&self.p);
        let (_s1, b1) = cc.overflowing_sub(&b0.as_());
        // if b1 == 1: return z
        // else:       return s0
        let mask = W::ZERO.wrapping_sub(&b1.as_());
        (z & mask) | (s0 & !mask)
    }

    /// Modular exponentiation, i.e., `x^exp (mod p)` where `p` is
    /// the modulus. Note that the runtime of this algorithm is
    /// linear in the bit length of `exp`.
    pub(crate) fn pow(&self, x: W, exp: W) -> W {
        let mut t = self.roots[0];
        let end = W::BITS - exp.leading_zeros();
        for i in (0..end).rev() {
            t = self.mul(t, t);
            if (exp >> i) & W::ONE != W::ZERO {
                t = self.mul(t, x);
            }
        }
        t
    }

    /// Modular inversion, i.e., x^-1 (mod p) where `p` is the
    /// modulus. Note that the runtime of this algorithm is
    /// linear in the bit length of `p`.
    #[inline(always)]
    pub(crate) fn inv(&self, x: W) -> W {
        self.pow(x, self.p - W::ONE - W::ONE)
    }

    /// Negation, i.e., `-x (mod p)` where `p` is the modulus.
    #[inline(always)]
    pub(crate) fn neg(&self, x: W) -> W {
        self.sub(W::ZERO, x)
    }

    /// Maps an integer to its internal representation. Field
    /// elements are mapped to the Montgomery domain in order to
    /// carry out field arithmetic. The result will be in [0, p).
    #[inline(always)]
    pub(crate) fn montgomery(&self, x: W) -> W {
        self.modp(self.mul(x, self.r2))
    }

    /// Maps a field element to its representation as an integer.
    /// The result will be in [0, p).
    #[inline(always)]
    pub(crate) fn residue(&self, x: W) -> W {
        self.modp(self.mul(x, W::ONE))
    }
}

impl Word for u32 {
    type Wide = u64;

    #[inline(always)]
    fn leading_zeros(self) -> u32 {
        u32::leading_zeros(self)
    }
}

impl Word for u64 {
    type Wide = u128;

    #[inline(always)]
    fn leading_zeros(self) -> u32 {
        u64::leading_zeros(self)
    }
}

macro_rules! impl_test_field_parameters_for {
    ($W: ident) => {
        #[cfg(test)]
        impl $crate::fp::tests::TestFieldParameters for FieldParametersSingleWord<$W> {
            fn p(&self) -> u128 {
                self.p.as_()
            }

            fn g(&self) -> u128 {
                self.g.as_()
            }

            fn base(&self) -> u128 {
                1u128 << <$W>::BITS
            }

            fn r2(&self) -> u128 {
                self.r2.as_()
            }

            fn mu(&self) -> u64 {
                self.mu as u64
            }

            fn bit_mask(&self) -> u128 {
                self.bit_mask.as_()
            }

            fn num_roots(&self) -> usize {
                self.num_roots
            }

            fn roots(&self) -> Vec<u128> {
                self.roots.map(AsPrimitive::as_).to_vec()
            }

            fn montgomery(&self, x: u128) -> u128 {
                Self::montgomery(self, x.try_into().unwrap()).into()
            }

            fn residue(&self, x: u128) -> u128 {
                Self::residue(self, x.try_into().unwrap()).into()
            }

            fn add(&self, x: u128, y: u128) -> u128 {
                Self::add(self, x.try_into().unwrap(), y.try_into().unwrap()).into()
            }

            fn sub(&self, x: u128, y: u128) -> u128 {
                Self::sub(self, x.try_into().unwrap(), y.try_into().unwrap()).into()
            }
            fn neg(&self, x: u128) -> u128 {
                Self::neg(self, x.try_into().unwrap()).into()
            }

            fn mul(&self, x: u128, y: u128) -> u128 {
                Self::mul(self, x.try_into().unwrap(), y.try_into().unwrap()).into()
            }

            fn pow(&self, x: u128, exp: u128) -> u128 {
                Self::pow(self, x.try_into().unwrap(), exp.try_into().unwrap()).into()
            }

            fn inv(&self, x: u128) -> u128 {
                Self::inv(self, x.try_into().unwrap()).into()
            }

            fn radix(&self) -> num_bigint::BigInt {
                num_bigint::BigInt::from(self.base())
            }
        }
    };
}

impl_test_field_parameters_for!(u32);
impl_test_field_parameters_for!(u64);
