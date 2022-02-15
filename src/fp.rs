// SPDX-License-Identifier: MPL-2.0

//! Finite field arithmetic for any field GF(p) for which p < 2^128.

#[cfg(test)]
use rand::{prelude::*, Rng};

use crate::expander::{get_expander, ExpID, XofID};

/// For each set of field parameters we pre-compute the 1st, 2nd, 4th, ..., 2^20-th principal roots
/// of unity. The largest of these is used to run the FFT algorithm on an input of size 2^20. This
/// is the largest input size we would ever need for the cryptographic applications in this crate.
pub(crate) const MAX_ROOTS: usize = 20;

/// This structure represents the parameters of a finite field GF(p) for which p < 2^128.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FieldParameters {
    /// The prime modulus `p`.
    pub p: u128,
    /// The number of bits of prime modulus `p`.
    pub bits: usize,
    /// `mu_montgomery = -p^(-1) mod 2^64`.
    pub mu_montgomery: u64,
    /// mu_barret = floor( (2^32)^(2*k) / p ), where k = ceil(self.bits/32).
    pub mu_barret: [u32; 5],
    /// `r2 = (2^128)^2 mod p`.
    pub r2: u128,
    /// The `2^num_roots`-th -principal root of unity. This element is used to generate the
    /// elements of `roots`.
    pub g: u128,
    /// The number of principal roots of unity in `roots`.
    pub num_roots: usize,
    /// Equal to `2^b - 1`, where `b` is the length of `p` in bits.
    pub bit_mask: u128,
    /// `roots[l]` is the `2^l`-th principal root of unity, i.e., `roots[l]` has order `2^l` in the
    /// multiplicative group. `root[l]` is equal to one by definition.
    pub roots: [u128; MAX_ROOTS + 1],
}

impl FieldParameters {
    /// Addition.
    pub fn add(&self, x: u128, y: u128) -> u128 {
        //   0,x
        // + 0,y
        // =====
        //   c,z
        let (z, carry) = x.overflowing_add(y);
        //     c, z
        // -   0, p
        // ========
        // b1,s1,s0
        let (s0, b0) = z.overflowing_sub(self.p);
        let (_s1, b1) = (carry as u128).overflowing_sub(b0 as u128);
        // if b1 == 1: return z
        // else:       return s0
        let m = 0u128.wrapping_sub(b1 as u128);
        (z & m) | (s0 & !m)
    }

    /// Subtraction.
    pub fn sub(&self, x: u128, y: u128) -> u128 {
        //     0, x
        // -   0, y
        // ========
        // b1,z1,z0
        let (z0, b0) = x.overflowing_sub(y);
        let (_z1, b1) = 0u128.overflowing_sub(b0 as u128);
        let m = 0u128.wrapping_sub(b1 as u128);
        //   z1,z0
        // +  0, p
        // ========
        //   s1,s0
        z0.wrapping_add(m & self.p)
    }

    /// Multiplication of field elements in the Montgomery domain. This uses the REDC algorithm
    /// described
    /// [here](https://www.ams.org/journals/mcom/1985-44-170/S0025-5718-1985-0777282-X/S0025-5718-1985-0777282-X.pdf).
    ///
    /// Example usage:
    /// assert_eq!(fp.from_elem(fp.mul(fp.elem(23), fp.elem(2))), 46);
    pub fn mul(&self, x: u128, y: u128) -> u128 {
        let x = [lo64(x), hi64(x)];
        let y = [lo64(y), hi64(y)];
        let p = [lo64(self.p), hi64(self.p)];
        let mut zz = [0; 4];
        let mut result: u128;
        let mut carry: u128;
        let mut hi: u128;
        let mut lo: u128;
        let mut cc: u128;

        // Integer multiplication
        // z = x * y

        //       x1,x0
        // *     y1,y0
        // ===========
        // z3,z2,z1,z0
        result = x[0] * y[0];
        carry = hi64(result);
        zz[0] = lo64(result);
        result = x[0] * y[1];
        hi = hi64(result);
        lo = lo64(result);
        result = lo + carry;
        zz[1] = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        zz[2] = lo64(result);

        result = x[1] * y[0];
        hi = hi64(result);
        lo = lo64(result);
        result = zz[1] + lo;
        zz[1] = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        carry = lo64(result);

        result = x[1] * y[1];
        hi = hi64(result);
        lo = lo64(result);
        result = lo + carry;
        lo = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        hi = lo64(result);
        result = zz[2] + lo;
        zz[2] = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        zz[3] = lo64(result);

        // Montgomery Reduction
        // z = z + p * mu*(z mod 2^64), where mu = (-p)^(-1) mod 2^64.

        // z3,z2,z1,z0
        // +     p1,p0
        // *         w = mu*z0
        // ===========
        // z3,z2,z1, 0
        let w = self.mu_montgomery.wrapping_mul(zz[0] as u64);
        result = p[0] * (w as u128);
        hi = hi64(result);
        lo = lo64(result);
        result = zz[0] + lo;
        zz[0] = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        carry = lo64(result);

        result = p[1] * (w as u128);
        hi = hi64(result);
        lo = lo64(result);
        result = lo + carry;
        lo = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        hi = lo64(result);
        result = zz[1] + lo;
        zz[1] = lo64(result);
        cc = hi64(result);
        result = zz[2] + hi + cc;
        zz[2] = lo64(result);
        cc = hi64(result);
        result = zz[3] + cc;
        zz[3] = lo64(result);

        //    z3,z2,z1
        // +     p1,p0
        // *         w = mu*z1
        // ===========
        //    z3,z2, 0
        let w = self.mu_montgomery.wrapping_mul(zz[1] as u64);
        result = p[0] * (w as u128);
        hi = hi64(result);
        lo = lo64(result);
        result = zz[1] + lo;
        zz[1] = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        carry = lo64(result);

        result = p[1] * (w as u128);
        hi = hi64(result);
        lo = lo64(result);
        result = lo + carry;
        lo = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        hi = lo64(result);
        result = zz[2] + lo;
        zz[2] = lo64(result);
        cc = hi64(result);
        result = zz[3] + hi + cc;
        zz[3] = lo64(result);
        cc = hi64(result);

        // z = (z3,z2)
        let prod = zz[2] | (zz[3] << 64);

        // Final subtraction
        // If z >= p, then z = z - p

        //     0, z
        // -   0, p
        // ========
        // b1,s1,s0
        let (s0, b0) = prod.overflowing_sub(self.p);
        let (_s1, b1) = (cc as u128).overflowing_sub(b0 as u128);
        // if b1 == 1: return z
        // else:       return s0
        let mask = 0u128.wrapping_sub(b1 as u128);
        (prod & mask) | (s0 & !mask)
    }

    /// Modular exponentiation, i.e., `x^exp (mod p)` where `p` is the modulus. Note that the
    /// runtime of this algorithm is linear in the bit length of `exp`.
    pub fn pow(&self, x: u128, exp: u128) -> u128 {
        let mut t = self.elem(1);
        for i in (0..128 - exp.leading_zeros()).rev() {
            t = self.mul(t, t);
            if (exp >> i) & 1 != 0 {
                t = self.mul(t, x);
            }
        }
        t
    }

    /// Modular inversion, i.e., x^-1 (mod p) where `p` is the modulus. Note that the runtime of
    /// this algorithm is linear in the bit length of `p`.
    pub fn inv(&self, x: u128) -> u128 {
        self.pow(x, self.p - 2)
    }

    /// Negation, i.e., `-x (mod p)` where `p` is the modulus.
    pub fn neg(&self, x: u128) -> u128 {
        self.sub(0, x)
    }

    /// Maps an integer to its internal representation. Field elements are mapped to the Montgomery
    /// domain in order to carry out field arithmetic.
    ///
    /// Example usage:
    /// let integer = 1; // Standard integer representation
    /// let elem = fp.elem(integer); // Internal representation in the Montgomery domain
    /// assert_eq!(elem, 2564090464);
    pub fn elem(&self, x: u128) -> u128 {
        modp(self.mul(x, self.r2), self.p)
    }

    /// Creates a field element from an arbitrary-length byte array in big endian order.
    pub fn from_be_bytes(&self, bytes: &[u8]) -> u128 {
        use std::convert::TryInto;
        let k = (self.bits + 31) / 32;
        let u32size = std::mem::size_of::<u32>();
        let u32words = (bytes.len() + u32size - 1) / u32size;
        let num_chunks = (u32words + k - 1) / k;
        let mut input = vec![0u32; k * num_chunks];

        for (i, ch) in bytes.rchunks(u32size).enumerate() {
            let mut chunk = vec![0u8; 4 - ch.len()];
            chunk.extend_from_slice(ch);
            input[i] = u32::from_be_bytes(chunk.try_into().unwrap());
        }

        let mut l = input.len();
        while l >= 2 * k {
            let chunk = &mut input[l - 2 * k..l];
            let reduced = self.barret(chunk);
            chunk[..k].copy_from_slice(&reduced);
            input.truncate(l - k);
            l = input.len();
        }

        let mut out = 0u128;
        for &i in input.iter().rev() {
            out = (out << 32) + i as u128;
        }
        out
    }

    /// Barret modular reduction using base b=2^32.
    ///
    /// Given a little-endian vector `x` of size 2*k, returns a little-endian vector corresponding
    /// to x mod p.
    /// Implementation follows Algorithm 14.42 - Barrett modular reduction as appears in
    /// "Handbook of Applied Cryptography", by A. Menezes, P. van Oorschot, and S. Vanstone.
    fn barret(&self, x: &[u32]) -> Vec<u32> {
        let k = (self.bits + 31) / 32;
        if x.len() < 2 * k {
            panic!("short input on barret reduction");
        }

        let mu = &self.mu_barret[..k + 1];
        let p64 = &[lo64(self.p) as u64, hi64(self.p) as u64];
        let p32 = &[
            lo32(p64[0]) as u32,
            hi32(p64[0]) as u32,
            lo32(p64[1]) as u32,
            hi32(p64[1]) as u32,
            0u32,
        ];
        let q1 = &x[k - 1..];
        let q2 = mul(q1, mu);
        let q3 = &q2[k + 1..];

        let r1 = &x[..k + 1];
        let r2m = mul(q3, &p32[..k]);
        let r2 = &r2m[..k + 1];
        let (mut r, is_r_neg) = sub(r1, r2);
        r = strip(r, k + 1);

        if is_r_neg {
            let mut bk1 = vec![0u32; k + 1];
            bk1.push(1);
            r = add(&r, &bk1); // r += b^(k+1)
            r.resize(k + 1, 0);
        }

        loop {
            let (r1, is_neg) = sub(&r, &p32[..k + 1]);
            if is_neg {
                r = strip(r, k);
                break;
            } else {
                r = strip(r1, k + 1);
            }
        }
        r
    }

    pub fn hash_to_field(&self, msg: &[u8], count: usize) -> Vec<u128> {
        let k = self.bits;
        let ell = (self.bits + k + 7) / 8;
        let length = count * ell;
        let dst = "Prio::DeriveFieldElements::DST".as_bytes();
        let exp = get_expander(ExpID::XOF(XofID::SHAKE128), dst, k);
        let pseudo = exp.expand(msg, length);
        let mut u = Vec::<u128>::with_capacity(count);
        for i in 0..count {
            let offset: usize = ell * i;
            let t = &pseudo[offset..(offset + ell)];
            u.push(self.from_be_bytes(t))
        }
        u
    }

    /// Returns a random field element mapped.
    #[cfg(test)]
    pub fn rand_elem<R: Rng + ?Sized>(&self, rng: &mut R) -> u128 {
        let uniform = rand::distributions::Uniform::from(0..self.p);
        self.elem(uniform.sample(rng))
    }

    /// Maps a field element to its representation as an integer.
    ///
    /// Example usage:
    /// let elem = 2564090464; // Internal representation in the Montgomery domain
    /// let integer = fp.from_elem(elem); // Standard integer representation
    /// assert_eq!(integer, 1);
    pub fn from_elem(&self, x: u128) -> u128 {
        modp(self.mul(x, 1), self.p)
    }

    #[cfg(test)]
    pub fn check(&self, p: u128, g: u128, order: u128) {
        use modinverse::modinverse;
        use num_bigint::{BigInt, ToBigInt};
        use std::cmp::max;

        assert_eq!(self.p, p, "p mismatch");

        let mu_montgomery = match modinverse((-(p as i128)).rem_euclid(1 << 64), 1 << 64) {
            Some(mu) => mu as u64,
            None => panic!("inverse of -p (mod 2^64) is undefined"),
        };
        assert_eq!(self.mu_montgomery, mu_montgomery, "mu_montgomery mismatch");

        let big_p = &p.to_bigint().unwrap();
        let k = (self.bits + 31) / 32;
        let big_mu_barret = (BigInt::from(1) << 32usize).pow(2 * k as u32) / big_p;
        let (_, mut mu_barret) = big_mu_barret.to_u32_digits();
        mu_barret.extend_from_slice(&vec![0u32; 5 - mu_barret.len()]);
        assert_eq!(self.mu_barret.to_vec(), mu_barret, "mu_barret mismatch");

        let big_r: &BigInt = &(&(BigInt::from(1) << 128) % big_p);
        let big_r2: &BigInt = &(&(big_r * big_r) % big_p);
        let mut it = big_r2.iter_u64_digits();
        let mut r2 = 0;
        r2 |= it.next().unwrap() as u128;
        if let Some(x) = it.next() {
            r2 |= (x as u128) << 64;
        }
        assert_eq!(self.r2, r2, "r2 mismatch");

        assert_eq!(self.g, self.elem(g), "g mismatch");
        assert_eq!(
            self.from_elem(self.pow(self.g, order)),
            1,
            "g order incorrect"
        );

        let num_roots = log2(order) as usize;
        assert_eq!(order, 1 << num_roots, "order not a power of 2");
        assert_eq!(self.num_roots, num_roots, "num_roots mismatch");

        let mut roots = vec![0; max(num_roots, MAX_ROOTS) + 1];
        roots[num_roots] = self.elem(g);
        for i in (0..num_roots).rev() {
            roots[i] = self.mul(roots[i + 1], roots[i + 1]);
        }
        assert_eq!(&self.roots, &roots[..MAX_ROOTS + 1], "roots mismatch");
        assert_eq!(self.from_elem(self.roots[0]), 1, "first root is not one");

        let bit_mask = (BigInt::from(1) << big_p.bits()) - BigInt::from(1);
        assert_eq!(
            self.bit_mask.to_bigint().unwrap(),
            bit_mask,
            "bit_mask mismatch"
        );
    }
}

// fn print_be_bytes(v: &[u8]) -> String {
//     let width = std::mem::size_of::<u8>() * 2; // 2 nibbles per byte
//     let mut s = String::new();
//     s += format!("len: {} data: 0x", v.len()).as_str();
//     for i in v.iter() {
//         s += format!("{:0width$x}", i, width = width).as_str();
//     }
//     s
// }
//
// fn print_vec<T: std::fmt::LowerHex>(v: &[T]) -> String {
//     let width = std::mem::size_of::<T>() * 2; // 2 nibbles per byte
//     let mut s = String::new();
//     s += format!("len: {} data: 0x", v.len()).as_str();
//     for i in v.iter().rev() {
//         s += format!("{:0width$x}", i, width = width).as_str();
//     }
//     s
// }

fn lo64(x: u128) -> u128 {
    x & ((1 << 64) - 1)
}

fn hi64(x: u128) -> u128 {
    x >> 64
}

fn lo32(x: u64) -> u64 {
    x & ((1 << 32) - 1)
}

fn hi32(x: u64) -> u64 {
    x >> 32
}

fn mul(x: &[u32], y: &[u32]) -> Vec<u32> {
    let l = x.len() + y.len();
    let mut z = vec![0u32; l];
    for (i, xi) in x.iter().enumerate() {
        let mut carry = 0;
        for (j, yj) in y.iter().enumerate() {
            let zij = ((*xi as u64) * (*yj as u64))
                .wrapping_add(z[i + j] as u64)
                .wrapping_add(carry);
            z[i + j] = lo32(zij) as u32;
            carry = hi32(zij);
        }
        z[i + y.len()] += lo32(carry) as u32;
    }
    z
}

fn carrying_add(x: u32, y: u32, carry_in: bool) -> (u32, bool) {
    let add = (x as u64) + (y as u64) + (carry_in as u64);
    let z = (add & 0xffffffff) as u32;
    let carry_out = ((add >> 32) != 0) as bool;
    (z, carry_out)
}

fn borrowing_sub(x: u32, y: u32, borrow_in: bool) -> (u32, bool) {
    let sub = (x as i64) - (y as i64) - (borrow_in as i64);
    let z = (sub & 0xffffffff) as u32;
    let borrow_out = ((sub >> 32) != 0) as bool;
    (z, borrow_out)
}

fn strip(mut x: Vec<u32>, k: usize) -> Vec<u32> {
    while let Some(&xi) = x.last() {
        if x.len() > k && xi == 0 {
            x.pop();
        } else {
            break;
        }
    }
    x
}

fn add(x: &[u32], y: &[u32]) -> Vec<u32> {
    if x.len() != y.len() {
        panic!("wrong sizes add: {} {}", x.len(), y.len())
    }
    let mut z = Vec::<u32>::with_capacity(x.len() + 1);
    let mut ci = false;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let (zi, co) = carrying_add(*xi, *yi, ci);
        z.push(zi);
        ci = co;
    }
    z.push(ci as u32);
    z
}

fn sub(x: &[u32], y: &[u32]) -> (Vec<u32>, bool) {
    if x.len() != y.len() {
        panic!("wrong sizes sub: {} {}", x.len(), y.len())
    }
    let mut z = Vec::<u32>::with_capacity(x.len() + 1);
    let mut bi = false;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let (zi, bo) = borrowing_sub(*xi, *yi, bi);
        z.push(zi);
        bi = bo;
    }
    z.push((0i32 - (bi as i32)) as u32);
    (z, bi)
}

fn modp(x: u128, p: u128) -> u128 {
    let (z, carry) = x.overflowing_sub(p);
    let m = 0u128.wrapping_sub(carry as u128);
    z.wrapping_add(m & p)
}

pub(crate) const FP32: FieldParameters = FieldParameters {
    p: 4293918721, // 32-bit prime
    bits: 32,
    mu_montgomery: 17302828673139736575,
    mu_barret: [1048831, 1, 0, 0, 0],
    r2: 1676699750,
    g: 1074114499,
    num_roots: 20,
    bit_mask: 4294967295,
    roots: [
        2564090464, 1729828257, 306605458, 2294308040, 1648889905, 57098624, 2788941825,
        2779858277, 368200145, 2760217336, 594450960, 4255832533, 1372848488, 721329415,
        3873251478, 1134002069, 7138597, 2004587313, 2989350643, 725214187, 1074114499,
    ],
};

pub(crate) const FP64: FieldParameters = FieldParameters {
    p: 18446744069414584321, // 64-bit prime
    bits: 64,
    mu_montgomery: 18446744069414584319,
    mu_barret: [4294967295, 0, 1, 0, 0],
    r2: 4294967295,
    g: 959634606461954525,
    num_roots: 32,
    bit_mask: 18446744073709551615,
    roots: [
        18446744065119617025,
        4294967296,
        18446462594437939201,
        72057594037927936,
        1152921504338411520,
        16384,
        18446743519658770561,
        18446735273187346433,
        6519596376689022014,
        9996039020351967275,
        15452408553935940313,
        15855629130643256449,
        8619522106083987867,
        13036116919365988132,
        1033106119984023956,
        16593078884869787648,
        16980581328500004402,
        12245796497946355434,
        8709441440702798460,
        8611358103550827629,
        8120528636261052110,
    ],
};

pub(crate) const FP96: FieldParameters = FieldParameters {
    p: 79228148845226978974766202881, // 96-bit prime
    bits: 96,
    mu_montgomery: 18446744073709551615,
    mu_barret: [406869090, 549081, 741, 1, 0],
    r2: 69162923446439011319006025217,
    g: 11329412859948499305522312170,
    num_roots: 64,
    bit_mask: 79228162514264337593543950335,
    roots: [
        10128756682736510015896859,
        79218020088544242464750306022,
        9188608122889034248261485869,
        10170869429050723924726258983,
        36379376833245035199462139324,
        20898601228930800484072244511,
        2845758484723985721473442509,
        71302585629145191158180162028,
        76552499132904394167108068662,
        48651998692455360626769616967,
        36570983454832589044179852640,
        72716740645782532591407744342,
        73296872548531908678227377531,
        14831293153408122430659535205,
        61540280632476003580389854060,
        42256269782069635955059793151,
        51673352890110285959979141934,
        43102967204983216507957944322,
        3990455111079735553382399289,
        68042997008257313116433801954,
        44344622755749285146379045633,
    ],
};

pub(crate) const FP128: FieldParameters = FieldParameters {
    p: 340282366920938462946865773367900766209, // 128-bit prime
    bits: 128,
    mu_montgomery: 18446744073709551615,
    mu_barret: [783, 0, 28, 0, 1],
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

// Compute the ceiling of the base-2 logarithm of `x`.
pub(crate) fn log2(x: u128) -> u128 {
    let y = (127 - x.leading_zeros()) as u128;
    y + ((x > 1 << y) as u128)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::{BigInt, Sign, ToBigInt};

    #[test]
    fn test_log2() {
        assert_eq!(log2(1), 0);
        assert_eq!(log2(2), 1);
        assert_eq!(log2(3), 2);
        assert_eq!(log2(4), 2);
        assert_eq!(log2(15), 4);
        assert_eq!(log2(16), 4);
        assert_eq!(log2(30), 5);
        assert_eq!(log2(32), 5);
        assert_eq!(log2(1 << 127), 127);
        assert_eq!(log2((1 << 127) + 13), 128);
    }

    struct TestFieldParametersData {
        fp: FieldParameters,  // The paramters being tested
        expected_p: u128,     // Expected fp.p
        expected_g: u128,     // Expected fp.from_elem(fp.g)
        expected_order: u128, // Expect fp.from_elem(fp.pow(fp.g, expected_order)) == 1
    }

    #[test]
    fn test_fp() {
        let test_fps = vec![
            TestFieldParametersData {
                fp: FP32,
                expected_p: 4293918721,
                expected_g: 3925978153,
                expected_order: 1 << 20,
            },
            TestFieldParametersData {
                fp: FP64,
                expected_p: 18446744069414584321,
                expected_g: 1753635133440165772,
                expected_order: 1 << 32,
            },
            TestFieldParametersData {
                fp: FP96,
                expected_p: 79228148845226978974766202881,
                expected_g: 34233996298771126927060021012,
                expected_order: 1 << 64,
            },
            TestFieldParametersData {
                fp: FP128,
                expected_p: 340282366920938462946865773367900766209,
                expected_g: 145091266659756586618791329697897684742,
                expected_order: 1 << 66,
            },
        ];

        for t in test_fps.into_iter().rev() {
            //  Check that the field parameters have been constructed properly.
            t.fp.check(t.expected_p, t.expected_g, t.expected_order);

            // Check that the generator has the correct order.
            assert_eq!(t.fp.from_elem(t.fp.pow(t.fp.g, t.expected_order)), 1);

            // Test arithmetic using the field parameters.
            arithmetic_test(&t.fp);
            barret_test(&t.fp);
            from_be_bytes_test(&t.fp);
        }
    }

    fn arithmetic_test(fp: &FieldParameters) {
        let mut rng = rand::thread_rng();
        let big_p = &fp.p.to_bigint().unwrap();

        for _ in 0..100 {
            let x = fp.rand_elem(&mut rng);
            let y = fp.rand_elem(&mut rng);
            let big_x = &fp.from_elem(x).to_bigint().unwrap();
            let big_y = &fp.from_elem(y).to_bigint().unwrap();

            // Test addition.
            let got = fp.add(x, y);
            let want = (big_x + big_y) % big_p;
            assert_eq!(fp.from_elem(got).to_bigint().unwrap(), want);

            // Test subtraction.
            let got = fp.sub(x, y);
            let want = if big_x >= big_y {
                big_x - big_y
            } else {
                big_p - big_y + big_x
            };
            assert_eq!(fp.from_elem(got).to_bigint().unwrap(), want);

            // Test multiplication.
            let got = fp.mul(x, y);
            let want = (big_x * big_y) % big_p;
            assert_eq!(fp.from_elem(got).to_bigint().unwrap(), want);

            // Test inversion.
            let got = fp.inv(x);
            let want = big_x.modpow(&(big_p - 2u128), big_p);
            assert_eq!(fp.from_elem(got).to_bigint().unwrap(), want);
            assert_eq!(fp.from_elem(fp.mul(got, x)), 1);

            // Test negation.
            let got = fp.neg(x);
            let want = (big_p - big_x) % big_p;
            assert_eq!(fp.from_elem(got).to_bigint().unwrap(), want);
            assert_eq!(fp.from_elem(fp.add(got, x)), 0);
        }
    }

    fn barret_test(fp: &FieldParameters) {
        let mut rng = rand::thread_rng();
        let k = (fp.bits + 31) / 32;
        let mut buf = vec![0u32; 2 * k];
        let mut big_buf: BigInt;
        let big_p = &fp.p.to_bigint().unwrap();
        let mut bytes = [0u8; 4];

        for _ in 1..1000 {
            big_buf = BigInt::default();
            for b in buf.iter_mut().rev() {
                rng.fill(&mut bytes);
                *b = u32::from_be_bytes(bytes);
                big_buf = (big_buf << 32) + BigInt::from_bytes_be(Sign::Plus, &bytes);
            }

            let buf_mod_p = fp.barret(&buf);
            let mut got = BigInt::default();
            for &i in buf_mod_p.iter().rev() {
                got = (got << 32) + i.to_bigint().unwrap();
            }
            let want = &big_buf % big_p;
            assert_eq!(got, want, "prime: {} input: {}", fp.p, big_buf);
        }
    }

    fn from_be_bytes_test(fp: &FieldParameters) {
        let mut rng = rand::thread_rng();
        let big_p = &fp.p.to_bigint().unwrap();

        for i in 0..1000 {
            let mut bytes = vec![0u8; i];
            rng.fill(&mut bytes[..]);
            let big_buf = BigInt::from_bytes_be(Sign::Plus, &bytes);
            let fp_elem = fp.from_be_bytes(&bytes);
            let got = fp_elem.to_bigint().unwrap();
            let want = &big_buf % big_p;
            assert_eq!(got, want, "prime: {} input: {}", fp.p, big_buf);
        }
    }
}
