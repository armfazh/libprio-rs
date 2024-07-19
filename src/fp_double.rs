// SPDX-License-Identifier: MPL-2.0

//! Finite field arithmetic for any field GF(p) for which p < 2^W,
//! where W is a specified word size.

use fixed::traits::FixedBits;
use num_traits::{
    ops::overflowing::{OverflowingAdd, OverflowingSub},
    AsPrimitive, PrimInt, Unsigned, WrappingAdd, WrappingMul, WrappingSub,
};

use crate::fp::MAX_ROOTS;

pub(crate) trait Word:
    Unsigned
    + PrimInt
    + FixedBits
    + OverflowingAdd
    + OverflowingSub
    + WrappingAdd
    + WrappingSub
    + WrappingMul
    + From<Self::Half>
    + AsPrimitive<Self::Half>
where
    bool: AsPrimitive<Self> + AsPrimitive<Self::Half>,
{
    type Half: Unsigned + PrimInt + FixedBits + WrappingMul;
}

/// impl_word_for used to explicitly check that the datatypes for `Word` are congruent.
#[macro_export]
macro_rules! impl_word_for_half {
    ($W:ident, $H:ident) => {
        const _: () = assert!($W::BITS == 2 * $H::BITS);
        impl $crate::fp_double::Word for $W {
            type Half = $H;
        }
    };
}

/// This structure represents the parameters of a finite field GF(p) for which
/// the prime p < 2^W, and W is a specified word size.
///
/// See also [`FieldParameters`](crate::fp::FieldParameters).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FieldParametersDoubleWord<W>
where
    W: Word,
    bool: AsPrimitive<W> + AsPrimitive<W::Half>,
{
    /// The prime modulus `p`.
    pub p: W,
    /// `mu = -p^(-1) mod 2^W`.
    pub mu: W::Half,
    /// `r2 = (2^W)^2 mod p`.
    pub r2: W,
    /// The `2^num_roots`-th -principal root of unity. This element is used to generate the
    /// elements of `roots`.
    pub g: W,
    /// The number of principal roots of unity in `roots`.
    pub num_roots: usize,
    /// Equal to `2^b - 1`, where `b` is the length of `p` in bits.
    pub bit_mask: W,
    /// `roots[l]` is the `2^l`-th principal root of unity, i.e., `roots[l]` has order `2^l` in the
    /// multiplicative group. `roots[0]` is equal to one by definition.
    pub roots: [W; MAX_ROOTS + 1],
}

impl<W> FieldParametersDoubleWord<W>
where
    W: Word,
    bool: AsPrimitive<W> + AsPrimitive<W::Half>,
{
    /// Addition. The result will be in [0, p), so long as both x and y are as well.
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
        let m = W::zero().wrapping_sub(&b1.as_());
        (z & m) | (s0 & !m)
    }

    /// Subtraction. The result will be in [0, p), so long as both x and y are as well.
    #[inline(always)]
    pub(crate) fn sub(&self, x: W, y: W) -> W {
        //        x
        // -      y
        // ========
        //    b0,z0
        let (z0, b0) = x.overflowing_sub(&y);
        let m = W::zero().wrapping_sub(&b0.as_());
        //      z0
        // +     p
        // ========
        //   s1,s0
        z0.wrapping_add(&(m & self.p))
        // if b1 == 1: return s0
        // else:       return z0
    }

    /// Multiplication of field elements in the Montgomery domain. This uses the
    /// Montgomery's [REDC algorithm][montgomery]. The result will be in [0, p).
    ///
    /// [montgomery]: https://www.ams.org/journals/mcom/1985-44-170/S0025-5718-1985-0777282-X/S0025-5718-1985-0777282-X.pdf
    #[inline(always)]
    pub(crate) fn mul(&self, x: W, y: W) -> W {
        let hi_lo = |v: W| -> (W, W) {
            (
                (v >> W::Half::BITS),
                (v & <W as From<W::Half>>::from(W::Half::MAX)),
            )
        };

        let hi64 = |v: W| -> W { v >> W::Half::BITS };
        let lo64 = |v: W| -> W { v & <W as From<W::Half>>::from(W::Half::MAX) };

        let (x1, x0) = hi_lo(x);
        let (y1, y0) = hi_lo(y);

        // Integer multiplication
        // z = x * y

        //       x1,x0
        // *     y1,y0
        // ===========
        // z3,z2,z1,z0

        //       x1,x0
        // *        y0
        // ===========
        //       z1,z0
        //    z2,z1
        let mut c2;
        let mut z1z0 = x0 * y0;
        let mut z2z1 = x1 * y0;

        (z2z1, c2) = z2z1.overflowing_add(&hi64(z1z0));
        let mut z3: W = c2.as_();

        //       x1,x0
        // *     y1,_
        // ===========
        // r3,r2,r1,r0
        //    s2,s1
        // s3,s2
        let s2s1 = x0 * y1;
        (z2z1, c2) = z2z1.overflowing_add(&s2s1);
        z3 = z3.wrapping_add(&c2.as_());

        let s3s2 = x1 * y1;
        (z2z1, c2) = z2z1.overflowing_add(&(lo64(s3s2) << W::Half::BITS));
        z3 = z3.wrapping_add(&c2.as_());
        z3 = z3.wrapping_add(&hi64(s3s2));

        println!("{:x} {:x?} {:x?}", z3, hi_lo(z2z1), hi_lo(z1z0));

        // Montgomery Reduction
        // z = z + p * mu*(z mod 2^W), where mu = (-p)^(-1) mod 2^W.
        //   = z + p * w, where w = mu*z0
        let z0: W::Half = lo64(z1z0).as_();
        let w = <W as From<W::Half>>::from(self.mu.wrapping_mul(&z0));
        let (p1, p0) = hi_lo(self.p);
        let p0w = p0.wrapping_mul(&w);
        println!("w: {:x} pw: {:x}", w, p0w);

        let c1;
        (z1z0, c1) = z1z0.overflowing_add(&p0w);
        (z2z1, c2) = z2z1.overflowing_add(&c1.as_());
        z3 = z3.wrapping_add(&c2.as_());

        (z2z1, c2) = z2z1.overflowing_add(&hi64(z1z0));
        z3 = z3.wrapping_add(&c2.as_());
        println!("{:x} {:x?} {:x?}", z3, hi_lo(z2z1), hi_lo(z1z0));

        let p1w = p1.wrapping_mul(&w);
        println!("w: {:x} pw: {:x}", w, p1w);

        (z2z1, c2) = z2z1.overflowing_add(&p1w);
        z3 = z3.wrapping_add(&c2.as_());

        let z1: W::Half = lo64(z2z1).as_();
        let w = <W as From<W::Half>>::from(self.mu.wrapping_mul(&z1));
        let p0w = p0.wrapping_mul(&w);
        println!("w: {:x} pw: {:x}", w, p0w);

        (z2z1, c2) = z2z1.overflowing_add(&p0w);
        z3 = z3.wrapping_add(&c2.as_());

        println!("{:x} {:x?} {:x?}", z3, hi_lo(z2z1), hi_lo(z1z0));

        let p1w = p1.wrapping_mul(&w);
        println!("w: {:x} pw: {:x}", w, p1w);

        (z2z1, c2) = z2z1.overflowing_add(&(lo64(p1w) << W::Half::BITS));
        z3 = z3.wrapping_add(&c2.as_());
        z3 = z3.wrapping_add(&hi64(p1w));

        let cc = hi64(z3);
        let z2 = hi64(z2z1);

        println!("{:x} {:x?} {:x?}", z3, hi_lo(z2z1), hi_lo(z1z0));
        // z = (z3,z2)
        let prod: W = z2 | (z3 << W::Half::BITS);

        // Final subtraction
        // If z >= p, then z = z - p

        //    cc, z
        // -   0, p
        // ========
        // b1,s1,s0
        let (s0, b0) = prod.overflowing_sub(&self.p);
        let (_s1, b1) = cc.overflowing_sub(&b0.as_());
        // if b1 == 1: return z
        // else:       return s0
        let mask = W::zero().wrapping_sub(&b1.as_());
        (prod & mask) | (s0 & !mask)

        // println!("{:x} {:x} {:x} {:x}", z3, z2, z1, z0);

        // (z2z1, c2) = z2z1.overflowing_add(&hi64(z1z0));
        // z3 = z3.wrapping_add(&c2.as_());
        // z3 = z3.wrapping_add(&hi64(s3s2));

        // let mut r2r1 = p1 * w;
        // (r2r1, c2) = r2r1.overflowing_add(&hi64(r1r0));
        // z3,z2,z1,z0
        // +     r1,r0
        // +  r2,r1
        // ===========
        // z3,z2,z1, 0
        // let w = self.mu.wrapping_mul(zz[0] as u64);
        // result = p[0] * (w as u128);
        // hi = hi64(result);
        // lo = lo64(result);
        // result = zz[0] + lo;
        // zz[0] = lo64(result);
        // cc = hi64(result);
        // result = hi + cc;
        // carry = lo64(result);

        // result = p[1] * (w as u128);
        // hi = hi64(result);
        // lo = lo64(result);
        // result = lo + carry;
        // lo = lo64(result);
        // cc = hi64(result);
        // result = hi + cc;
        // hi = lo64(result);
        // result = zz[1] + lo;
        // zz[1] = lo64(result);
        // cc = hi64(result);
        // result = zz[2] + hi + cc;
        // zz[2] = lo64(result);
        // cc = hi64(result);
        // result = zz[3] + cc;
        // zz[3] = lo64(result);

        // //    z3,z2,z1
        // // +     p1,p0
        // // *         w = mu*z1
        // // ===========
        // //    z3,z2, 0
        // let w = self.mu.wrapping_mul(zz[1] as u64);
        // result = p[0] * (w as u128);
        // hi = hi64(result);
        // lo = lo64(result);
        // result = zz[1] + lo;
        // zz[1] = lo64(result);
        // cc = hi64(result);
        // result = hi + cc;
        // carry = lo64(result);

        // result = p[1] * (w as u128);
        // hi = hi64(result);
        // lo = lo64(result);
        // result = lo + carry;
        // lo = lo64(result);
        // cc = hi64(result);
        // result = hi + cc;
        // hi = lo64(result);
        // result = zz[2] + lo;
        // zz[2] = lo64(result);
        // cc = hi64(result);
        // result = zz[3] + hi + cc;
        // zz[3] = lo64(result);
        // cc = hi64(result);

        // // z = (z3,z2)
        // let prod = zz[2] | (zz[3] << 64);

        // // Final subtraction
        // // If z >= p, then z = z - p

        // //    cc, z
        // // -   0, p
        // // ========
        // // b1,s1,s0
        // let (s0, b0) = prod.overflowing_sub(self.p);
        // let (_s1, b1) = cc.overflowing_sub(b0 as u128);
        // // if b1 == 1: return z
        // // else:       return s0
        // let mask = 0u128.wrapping_sub(b1 as u128);
        // (prod & mask) | (s0 & !mask);

        // Integer multiplication
        // z = x * y

        //     x
        // *   y
        // =====
        // z1,z0
        // let (z1, z0) = hi_lo(W::Wide::from(x) * W::Wide::from(y));

        // // Montgomery Reduction
        // // z = z + p * mu*(z mod 2^W), where mu = (-p)^(-1) mod 2^W.
        // //   = z + p * w, where w = mu*z0
        // let w = self.mu.wrapping_mul(&z0);
        // let (r1, r0) = hi_lo(W::Wide::from(self.p) * W::Wide::from(w));

        // //    z1,z0
        // // +  r1,r0
        // //    =====
        // // cc, z, 0
        // let (_zero, carry) = z0.overflowing_add(&r0);
        // let (cc, z) =
        //     hi_lo(W::Wide::from(z1) + W::Wide::from(r1) + AsPrimitive::<W::Wide>::as_(carry));

        // // Final subtraction
        // // If z >= p, then z = z - p

        // //    cc, z
        // // -   0, p
        // // ========
        // // b1,s1,s0
        // let (s0, b0) = z.overflowing_sub(&self.p);
        // let (_s1, b1) = cc.overflowing_sub(&b0.as_());
        // // if b1 == 1: return z
        // // else:       return s0
        // let mask = W::zero().wrapping_sub(&b1.as_());
        // (z & mask) | (s0 & !mask)
    }

    /// Modular exponentiation, i.e., `x^exp (mod p)` where `p` is the modulus. Note that the
    /// runtime of this algorithm is linear in the bit length of `exp`.
    pub(crate) fn pow(&self, x: W, exp: W) -> W {
        let mut t = self.roots[0];
        for i in (0..W::BITS - exp.leading_zeros()).rev() {
            t = self.mul(t, t);
            if (exp >> i) & W::one() != W::zero() {
                t = self.mul(t, x);
            }
        }
        t
    }

    /// Modular inversion, i.e., x^-1 (mod p) where `p` is the modulus. Note that the runtime of
    /// this algorithm is linear in the bit length of `p`.
    #[inline(always)]
    pub(crate) fn inv(&self, x: W) -> W {
        self.pow(x, self.p - W::one() - W::one())
    }

    /// Negation, i.e., `-x (mod p)` where `p` is the modulus.
    #[inline(always)]
    pub(crate) fn neg(&self, x: W) -> W {
        self.sub(W::zero(), x)
    }

    /// Maps an integer to its internal representation. Field elements are mapped to the Montgomery
    /// domain in order to carry out field arithmetic. The result will be in [0, p).
    #[inline(always)]
    pub(crate) fn montgomery(&self, x: W) -> W {
        self.modp(self.mul(x, self.r2))
    }

    /// Maps a field element to its representation as an integer. The result will be in [0, p).
    #[inline(always)]
    pub(crate) fn residue(&self, x: W) -> W {
        self.modp(self.mul(x, W::one()))
    }

    #[inline(always)]
    fn modp(&self, x: W) -> W {
        let (z, carry) = x.overflowing_sub(&self.p);
        let m = W::zero().wrapping_sub(&carry.as_());
        z.wrapping_add(&(m & self.p))
    }
}

impl_word_for_half!(u128, u64);

pub(crate) const FP128: FieldParametersDoubleWord<u128> = FieldParametersDoubleWord {
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

#[cfg(test)]
mod tests {
    use super::FP128;

    #[test]
    fn devel() {
        let x = FP128.g;
        let y = FP128.g;
        let z = FP128.mul(x, y);

        println!("x: {:x}", x);
        println!("y: {:x}", y);
        println!("z: {:x}", z);
    }
}
