// Copyright (c) 2024 Cloudflare, Inc.
// SPDX-License-Identifier: MPL-2.0

//! Implementation of a Verifiable Incremental Distributed Point Function (VIDPF).
//!
//! The VIDPF construction is specified in [[draft-mouris-cfrg-mastic]] and builds
//! on techniques from [[MST23]] and [[CP22]] to lift an IDPF to a VIDPF.
//!
//! [CP22]: https://doi.org/10.1007/978-3-031-06944-4_6
//! [MST23]: https://eprint.iacr.org/2023/080
//! [draft-mouris-cfrg-mastic]: https://datatracker.ietf.org/doc/draft-mouris-cfrg-mastic/01/

use std::{
    borrow::Borrow,
    error::Error,
    iter::zip,
    marker::PhantomData,
    ops::{Add, BitAnd, BitXor, ControlFlow, Sub},
};

use rand_core::RngCore;
use subtle::{Choice, ConditionallyNegatable, ConditionallySelectable};

use crate::{
    codec::Decode,
    field::{FieldElement, FieldElementExt},
    vdaf::xof::{Seed as XofSeed, Xof},
};

/// Creates a new instance of a VIDPF.
pub fn new_vidpf<P, R>(sec_param: usize, prng_ctor: P, bind_info: Vec<u8>) -> VidpfInstance<P, R>
where
    P: GetPRNG,
    R: Range,
    for<'a> &'a R: RangeOps<R>,
{
    VidpfInstance {
        key_size: sec_param / 8,
        seed_size: sec_param / 8,
        proof_size: 2 * (sec_param / 8),
        prng_ctor,
        bind_info,
        _cd: PhantomData,
    }
}

/// VidpfInstance contains the global parameters of a VIDPF instance.
pub struct VidpfInstance<P, R>
where
    // Primitive to construct a pseudorandom generator.
    P: GetPRNG,
    // Represents the codomain of a Distributed Point Function (DPF).
    R: Range,
    for<'a> &'a R: RangeOps<R>,
{
    // Key size in bytes.
    key_size: usize,
    // Proof size in bytes.
    proof_size: usize,
    // Seed size in bytes.
    seed_size: usize,
    // Constructor of a seeded pseudorandom generator.
    prng_ctor: P,
    // Used to cryptographically bind some information.
    bind_info: Vec<u8>,

    _cd: PhantomData<R>,
}

impl<P, R> VidpfInstance<P, R>
where
    P: GetPRNG,
    R: Range,
    for<'a> &'a R: RangeOps<R>,
{
    /// The gen method splits the point function `F(alpha) = beta` in two private keys
    /// used by the aggregation servers, and a common public key.
    /// The beta value muist be a non-zero value.
    pub fn gen(&self, alpha: &[u8], beta: &R) -> Result<(PublicKey<R>, Key, Key), Box<dyn Error>> {
        assert!(!beta.is_zero(), "beta cannot be zero");

        // Key Generation.
        let k0 = Key::gen(ServerID::S0, self.key_size)?;
        let k1 = Key::gen(ServerID::S1, self.key_size)?;

        let mut s_i = [Seed::from(&k0), Seed::from(&k1)];
        let mut t_i = [ControlBit::from(&k0), ControlBit::from(&k1)];

        let n = 8 * alpha.len();
        let mut cw = Vec::with_capacity(n);
        let mut cs = Vec::with_capacity(n);

        for i in 0..n {
            let alpha_i = ControlBit::from((alpha[i / 8] >> (i % 8)) & 0x1);

            let Sequence(sl_0, tl_0, sr_0, tr_0) = self.prg(&s_i[0]);
            let Sequence(sl_1, tl_1, sr_1, tr_1) = self.prg(&s_i[1]);

            let s_same_0 = &Seed::conditional_select(&sl_0, &sr_0, !alpha_i.0);
            let s_same_1 = &Seed::conditional_select(&sl_1, &sr_1, !alpha_i.0);

            let s_cw = s_same_0 ^ s_same_1;
            let t_cw_l = tl_0 ^ tl_1 ^ alpha_i ^ ControlBit::from(1);
            let t_cw_r = tr_0 ^ tr_1 ^ alpha_i;
            let t_cw_diff = ControlBit::conditional_select(&t_cw_l, &t_cw_r, alpha_i.0);

            let s_diff_0 = &Seed::conditional_select(&sl_0, &sr_0, alpha_i.0);
            let s_diff_1 = &Seed::conditional_select(&sl_1, &sr_1, alpha_i.0);
            let s_tilde_i_0 = s_diff_0 ^ (t_i[0] & s_cw.borrow()).borrow();
            let s_tilde_i_1 = s_diff_1 ^ (t_i[1] & s_cw.borrow()).borrow();

            let t_diff_0 = ControlBit::conditional_select(&tl_0, &tr_0, alpha_i.0);
            let t_diff_1 = ControlBit::conditional_select(&tl_1, &tr_1, alpha_i.0);
            t_i[0] = t_diff_0 ^ (t_i[0] & t_cw_diff);
            t_i[1] = t_diff_1 ^ (t_i[1] & t_cw_diff);

            let (w_i_0, w_i_1): (R, R);
            (s_i[0], w_i_0) = self.convert(&s_tilde_i_0);
            (s_i[1], w_i_1) = self.convert(&s_tilde_i_1);

            let mut w_cw = (beta - w_i_0.borrow()).borrow() + w_i_1.borrow();
            w_cw.conditional_negate(t_i[1].0);

            let cw_i = CorrectionWord {
                s_cw,
                t_cw_l,
                t_cw_r,
                w_cw,
            };
            cw.push(cw_i);

            let pi_0 = &self.hash_one(alpha, i, &s_i[0]);
            let pi_1 = &self.hash_one(alpha, i, &s_i[1]);
            let pi_i = pi_0 ^ pi_1;
            cs.push(pi_i);
        }

        Ok((PublicKey { cw, cs }, k0, k1))
    }

    /// eval_next queries one bit `(alpha_i)` in the evaluation tree at the level
    /// corresponding to the current state `(s_i,t_i,cw_i)`. This evaluation updates
    /// the state (s_i,t_i) and returns a new secret-shared value `(y)`.
    pub fn eval_next(
        &self,
        b: ServerID,
        alpha_i: ControlBit,
        s_i: &mut Seed,
        t_i: &mut ControlBit,
        cw_i: &CorrectionWord<R>,
    ) -> R {
        let CorrectionWord {
            s_cw,
            t_cw_l,
            t_cw_r,
            w_cw,
        } = cw_i;

        let seq_tilde = &self.prg(s_i);
        let seq_cw = &Sequence(s_cw.clone(), *t_cw_l, s_cw.clone(), *t_cw_r);
        let Sequence(sl, tl, sr, tr) = seq_tilde ^ (*t_i & seq_cw).borrow();

        let s_tilde_i = Seed::conditional_select(&sl, &sr, alpha_i.0);
        *t_i = ControlBit::conditional_select(&tl, &tr, alpha_i.0);

        let w_i: R;
        (*s_i, w_i) = self.convert(&s_tilde_i);

        let mut y_i = R::conditional_select(&w_i, (&w_i + w_cw).borrow(), t_i.0);
        y_i.conditional_negate(Choice::from(b as u8));

        y_i
    }

    /// proof_next generates a new proof at the current level `(alpha,level)` from the
    /// current state `(s_i,t_i,cs_i)` and current proof `(pi)`.
    pub fn proof_next(
        &self,
        pi: &Proof,
        alpha: &[u8],
        level: usize,
        si: &Seed,
        ti: ControlBit,
        cs_i: &Proof,
    ) -> Proof {
        let pi_tilde = &self.hash_one(alpha, level, si);
        let h2_input = pi ^ (pi_tilde ^ (ti & cs_i).borrow()).borrow();
        let out_pi = pi ^ self.hash_two(&h2_input).borrow();
        out_pi
    }

    /// Eval queries the evaluation tree with input alpha and produces a share of the output value.
    pub fn eval(&self, alpha: &[u8], key: &Key, pk: &PublicKey<R>) -> Share<R> {
        assert!(key.value.len() == self.key_size, "bad key size");

        let n = 8 * alpha.len();
        assert!(pk.cw.len() >= n, "bad public key size of cw field");
        assert!(pk.cs.len() >= n, "bad public key size of cs field");

        let mut s_i = Seed::from(key);
        let mut t_i = ControlBit::from(key);

        let mut y = R::new();
        let mut pi = self.initial_proof();

        for i in 0..n {
            let alpha_i = ControlBit::from((alpha[i / 8] >> (i % 8)) & 0x1);
            y = self.eval_next(key.id, alpha_i, &mut s_i, &mut t_i, &pk.cw[i]);
            pi = self.proof_next(&pi, alpha, i, &s_i, t_i, &pk.cs[i]);
        }

        Share { y, pi }
    }

    /// verify checks whether the proofs are equal and that the shares add up to beta.
    pub fn verify(&self, a: &Share<R>, b: &Share<R>, beta: &R) -> bool {
        assert!(a.pi.0.len() == self.proof_size);
        assert!(b.pi.0.len() == self.proof_size);
        &a.y + &b.y == *beta && a.pi.0 == b.pi.0
    }

    /// initial_proof returns the initial value of a Proof for the root level.
    pub fn initial_proof(&self) -> Proof {
        Proof::new(self.proof_size)
    }

    /// initial_seed returns the initial value of a Seed for the root level.
    pub fn initial_seed(&self) -> Seed {
        Seed::new(self.seed_size)
    }

    fn prg(&self, seed: &Seed) -> Sequence {
        let dst = "100".as_bytes();
        let mut prg = self.prng_ctor.new_prng(seed, dst, &self.bind_info);

        let mut sl = self.initial_seed();
        let mut sr = self.initial_seed();
        let mut tl = ControlBit::from(0);
        let mut tr = ControlBit::from(0);
        sl.fill(&mut prg);
        sr.fill(&mut prg);
        tl.fill(&mut prg);
        tr.fill(&mut prg);

        Sequence(sl, tl, sr, tr)
    }

    fn convert(&self, seed: &Seed) -> (Seed, R) {
        let dst = "101".as_bytes();
        let mut prg = self.prng_ctor.new_prng(seed, dst, &self.bind_info);

        let mut out_seed = self.initial_seed();
        let mut value = R::new();
        out_seed.fill(&mut prg);
        value.fill(&mut prg);

        (out_seed, value)
    }

    fn hash_one(&self, alpha: &[u8], level: usize, seed: &Seed) -> Proof {
        let dst = "vidpf cs proof".as_bytes();
        let mut binder = Vec::new();
        binder.extend_from_slice(alpha);
        binder.extend(level.to_le_bytes());
        let mut prg = self.prng_ctor.new_prng(seed, dst, &binder);

        let mut proof = self.initial_proof();
        proof.fill(&mut prg);

        proof
    }

    fn hash_two(&self, proof: &Proof) -> Proof {
        let dst = "vidpf proof adjustment".as_bytes();
        let seed = self.initial_seed();
        let binder = &proof.0;
        let mut prg = self.prng_ctor.new_prng(&seed, dst, binder);

        let mut out_proof = self.initial_proof();
        out_proof.fill(&mut prg);

        out_proof
    }
}

#[derive(Debug, Clone, Copy)]
/// ServerID used to identify two aggregation servers.
pub enum ServerID {
    /// S0 is the first server.
    S0 = 0,
    /// S1 is the second server.
    S1 = 1,
}

#[derive(Debug)]
/// Share
pub struct Share<R>
where
    R: Range,
    for<'a> &'a R: RangeOps<R>,
{
    /// y is the sharing of the output.
    pub y: R,
    /// pi is a proof used to verify the share.
    pub pi: Proof,
}

/// RangeOps
pub trait RangeOps<R>: Sized + Add<Output = R> + Sub<Output = R> {}

/// Range
pub trait Range: Fill + PartialEq + ConditionallyNegatable + ConditionallySelectable
where
    for<'a> &'a Self: RangeOps<Self>,
{
    /// new
    fn new() -> Self;
    /// is_zero
    fn is_zero(self) -> bool;
}

/// Fill populates a struct from a PRNG source.
pub trait Fill {
    /// fill reads as many bytes as needed from a PRNG source to fill the [Self] struct.
    fn fill(&mut self, r: &mut Box<dyn RngCore>);
}

/// GetPRG
pub trait GetPRNG {
    /// new_prng
    fn new_prng(&self, seed: &Seed, dst: &[u8], binder: &[u8]) -> Box<dyn RngCore>;
}

/// PrngFromXof is a helper to create a PRNG from any [crate::vdaf::xof::Xof] implementer.
pub struct PrngFromXof<const SEED_SIZE: usize, X: Xof<SEED_SIZE>>(PhantomData<X>);

impl<const SEED_SIZE: usize, X: Xof<SEED_SIZE>> Default for PrngFromXof<SEED_SIZE, X> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<const SEED_SIZE: usize, X: Xof<SEED_SIZE>> GetPRNG for PrngFromXof<SEED_SIZE, X>
where
    <X as Xof<SEED_SIZE>>::SeedStream: 'static,
{
    fn new_prng(&self, seed: &Seed, dst: &[u8], binder: &[u8]) -> Box<dyn RngCore> {
        let xof_seed = XofSeed::<SEED_SIZE>::get_decoded(&seed.0).unwrap();
        Box::new(X::seed_stream(&xof_seed, dst, binder))
    }
}

#[derive(Debug)]
/// CorrectionWord
pub struct CorrectionWord<R>
where
    R: Range,
    for<'a> &'a R: RangeOps<R>,
{
    /// s_cw
    s_cw: Seed,
    /// t_cw_l
    t_cw_l: ControlBit,
    /// t_cw_r
    t_cw_r: ControlBit,
    /// w_cw
    w_cw: R,
}

#[derive(Debug, PartialEq)]
/// Proof
pub struct Proof(Vec<u8>);

impl Proof {
    /// new
    pub fn new(n: usize) -> Self {
        Self(vec![0; n])
    }
}

impl Fill for Proof {
    fn fill(&mut self, r: &mut Box<dyn RngCore>) {
        r.fill_bytes(&mut self.0)
    }
}

impl BitXor for &Proof {
    type Output = Proof;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Proof(zip(&self.0, &rhs.0).map(|(a, b)| a ^ b).collect())
    }
}

#[derive(Debug)]
/// PublicKey is used by aggregation servers.
pub struct PublicKey<R>
where
    R: Range,
    for<'a> &'a R: RangeOps<R>,
{
    cw: Vec<CorrectionWord<R>>,
    cs: Vec<Proof>,
}

#[derive(Debug)]
/// Key is the aggreagation server's private key.
pub struct Key {
    id: ServerID,
    value: Vec<u8>,
}

impl Key {
    /// generates a key of n bytes at random.
    pub fn gen(id: ServerID, n: usize) -> Result<Self, Box<dyn Error>> {
        let mut value = vec![0; n];
        getrandom::getrandom(&mut value)?;
        Ok(Key { id, value })
    }
}

struct Sequence(Seed, ControlBit, Seed, ControlBit);

impl BitXor for &Sequence {
    type Output = Sequence;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Sequence(
            &self.0 ^ &rhs.0,
            self.1 ^ rhs.1,
            &self.2 ^ &rhs.2,
            self.3 ^ rhs.3,
        )
    }
}

#[derive(Debug, Clone, Copy)]
/// ControlBit
pub struct ControlBit(Choice);

impl From<u8> for ControlBit {
    fn from(b: u8) -> Self {
        ControlBit(Choice::from(b))
    }
}

impl From<&Key> for ControlBit {
    fn from(k: &Key) -> Self {
        ControlBit::from(k.id as u8)
    }
}

impl Fill for ControlBit {
    fn fill(&mut self, r: &mut Box<dyn RngCore>) {
        let mut b = [0u8; 1];
        r.fill_bytes(&mut b);
        *self = ControlBit::from(b[0] & 0x1)
    }
}

impl ConditionallySelectable for ControlBit {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        ControlBit((a.0 & !choice) | (b.0 & choice))
    }
}

impl BitAnd<&Seed> for ControlBit {
    type Output = Seed;
    fn bitand(self, rhs: &Seed) -> Self::Output {
        Seed(
            rhs.0
                .iter()
                .map(|x| u8::conditional_select(&0, x, self.0))
                .collect(),
        )
    }
}

impl BitAnd<&Proof> for ControlBit {
    type Output = Proof;
    fn bitand(self, rhs: &Proof) -> Self::Output {
        Proof(
            rhs.0
                .iter()
                .map(|x| u8::conditional_select(&0, x, self.0))
                .collect(),
        )
    }
}

impl BitAnd<&Sequence> for ControlBit {
    type Output = Sequence;

    fn bitand(self, rhs: &Sequence) -> Self::Output {
        Sequence(self & &rhs.0, self & rhs.1, self & &rhs.2, self & rhs.3)
    }
}

impl BitAnd for ControlBit {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        ControlBit(self.0 & rhs.0)
    }
}

impl BitXor for ControlBit {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        ControlBit(self.0 ^ rhs.0)
    }
}

#[derive(Debug, Clone)]
/// Seed
pub struct Seed(Vec<u8>);

impl Seed {
    /// new
    pub fn new(n: usize) -> Self {
        Self(vec![0; n])
    }

    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Seed(
            zip(&a.0, &b.0)
                .map(|(a, b)| u8::conditional_select(a, b, choice))
                .collect(),
        )
    }
}

impl From<&Key> for Seed {
    fn from(k: &Key) -> Self {
        Seed(k.value.clone())
    }
}

impl Fill for Seed {
    fn fill(&mut self, r: &mut Box<(dyn RngCore + 'static)>) {
        r.fill_bytes(&mut self.0)
    }
}

impl BitXor for &Seed {
    type Output = Seed;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Seed(zip(&self.0, &rhs.0).map(|(a, b)| a ^ b).collect())
    }
}

#[derive(Debug, Clone, Copy)]
/// Weight
pub struct Weight<F: FieldElement, const N: usize>(pub [F; N]);

impl<F: FieldElement, const N: usize> RangeOps<Weight<F, N>> for &Weight<F, N> {}

impl<F: FieldElement, const N: usize> Range for Weight<F, N> {
    fn new() -> Self {
        Self([F::zero(); N])
    }
    fn is_zero(self) -> bool {
        self.0 == [F::zero(); N]
    }
}

impl<F: FieldElement, const N: usize> Fill for Weight<F, N> {
    fn fill(&mut self, r: &mut Box<dyn RngCore>) {
        let mut bytes = vec![0; F::ENCODED_SIZE];
        self.0.iter_mut().for_each(|i| loop {
            r.fill_bytes(&mut bytes);
            if let ControlFlow::Break(x) = F::from_random_rejection(&bytes) {
                *i = x;
                break;
            }
        });
    }
}

impl<F: FieldElement, const N: usize> ConditionallySelectable for Weight<F, N> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let mut out = Weight::<F, N>::new();
        zip(&mut out.0, zip(&a.0, &b.0))
            .for_each(|(c, (a, b))| *c = F::conditional_select(a, b, choice));
        out
    }
}
impl<F: FieldElement, const N: usize> ConditionallyNegatable for Weight<F, N> {
    fn conditional_negate(&mut self, choice: Choice) {
        self.0.iter_mut().for_each(|a| a.conditional_negate(choice));
    }
}

impl<F: FieldElement, const N: usize> PartialEq for Weight<F, N> {
    fn eq(&self, rhs: &Self) -> bool {
        N == self.0.len() && N == rhs.0.len() && zip(self.0, rhs.0).all(|(a, b)| a == b)
    }
}

impl<F: FieldElement, const N: usize> Add for &Weight<F, N> {
    type Output = Weight<F, N>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut out = Weight::<F, N>::new();
        zip(&mut out.0, zip(self.0, rhs.0)).for_each(|(c, (a, b))| *c = a + b);
        out
    }
}

impl<F: FieldElement, const N: usize> Sub for &Weight<F, N> {
    type Output = Weight<F, N>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = Weight::<F, N>::new();
        zip(&mut out.0, zip(self.0, rhs.0)).for_each(|(c, (a, b))| *c = a - b);
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        field::Field128,
        vdaf::xof::XofTurboShake128,
        vidpf::{new_vidpf, ControlBit, PrngFromXof, Range, Seed, VidpfInstance, Weight},
    };

    fn setup() -> VidpfInstance<PrngFromXof<16, XofTurboShake128>, Weight<Field128, 3>> {
        let prng = PrngFromXof::<16, XofTurboShake128>::default();
        const SEC_PARAM: usize = 128;
        let binder = "Mock Protocol uses a VIDPF".as_bytes().to_vec();
        new_vidpf(SEC_PARAM, prng, binder)
    }

    #[test]
    fn happy_path() {
        let vidpf = setup();
        let alpha: &[u8] = &[0xF];
        let beta = Weight([21.into(), 22.into(), 23.into()]);

        let (public, k0, k1) = vidpf.gen(alpha, &beta).unwrap();
        let share0 = vidpf.eval(alpha, &k0, &public);
        let share1 = vidpf.eval(alpha, &k1, &public);

        assert!(vidpf.verify(&share0, &share1, &beta), "verification failed")
    }

    #[test]
    fn bad_query() {
        let vidpf = setup();
        let alpha: &[u8] = &[0xF];
        let alpha_bad: &[u8] = &[0x0];
        let beta = Weight([21.into(), 22.into(), 23.into()]);

        let (public, k0, k1) = vidpf.gen(alpha, &beta).unwrap();
        let share0 = vidpf.eval(alpha_bad, &k0, &public);
        let share1 = vidpf.eval(alpha_bad, &k1, &public);

        assert!(
            &share0.y + &share1.y == Weight::new(),
            "shares must add up to zero"
        );
        assert!(
            !vidpf.verify(&share0, &share1, &beta),
            "verification passed, but it should failed"
        )
    }

    #[test]
    fn tree_property_correct_query() {
        let vidpf = setup();
        let alpha: &[u8] = &[0xF];
        let beta = Weight([21.into(), 22.into(), 23.into()]);

        let (public, k0, k1) = vidpf.gen(alpha, &beta).unwrap();

        let mut s_i_0 = Seed::from(&k0);
        let mut t_i_0 = ControlBit::from(&k0);
        let mut s_i_1 = Seed::from(&k1);
        let mut t_i_1 = ControlBit::from(&k1);
        let mut pi_0 = vidpf.initial_proof();
        let mut pi_1 = vidpf.initial_proof();

        for i in 0..alpha.len() {
            let alpha_i = ControlBit::from((alpha[i / 8] >> (i % 8)) & 0x1);
            let share_0 = vidpf.eval_next(k0.id, alpha_i, &mut s_i_0, &mut t_i_0, &public.cw[i]);
            let share_1 = vidpf.eval_next(k1.id, alpha_i, &mut s_i_1, &mut t_i_1, &public.cw[i]);

            assert!(
                &share_0 + &share_1 == beta,
                "shares must add up to beta at each level"
            );

            pi_0 = vidpf.proof_next(&pi_0, alpha, i, &s_i_0, t_i_0, &public.cs[i]);
            pi_1 = vidpf.proof_next(&pi_1, alpha, i, &s_i_0, t_i_0, &public.cs[i]);
            assert!(pi_0 == pi_1, "proofs must be equal at each level");
        }
    }

    #[test]
    fn tree_property_bad_query() {
        let vidpf = setup();
        let alpha: &[u8] = &[0xF];
        let alpha_bad: &[u8] = &[0x0];
        let beta = Weight([21.into(), 22.into(), 23.into()]);
        let zero = Weight::new();

        let (public, k0, k1) = vidpf.gen(alpha, &beta).unwrap();

        let mut s_i_0 = Seed::from(&k0);
        let mut t_i_0 = ControlBit::from(&k0);
        let mut s_i_1 = Seed::from(&k1);
        let mut t_i_1 = ControlBit::from(&k1);
        let mut pi_0 = vidpf.initial_proof();
        let mut pi_1 = vidpf.initial_proof();

        for i in 0..alpha_bad.len() {
            // it queries a bad input
            let alpha_i = ControlBit::from((alpha_bad[i / 8] >> (i % 8)) & 0x1);
            let share_0 = vidpf.eval_next(k0.id, alpha_i, &mut s_i_0, &mut t_i_0, &public.cw[i]);
            let share_1 = vidpf.eval_next(k1.id, alpha_i, &mut s_i_1, &mut t_i_1, &public.cw[i]);

            assert!(
                &share_0 + &share_1 == zero,
                "shares must add up to zero at each level"
            );

            pi_0 = vidpf.proof_next(&pi_0, alpha, i, &s_i_0, t_i_0, &public.cs[i]);
            pi_1 = vidpf.proof_next(&pi_1, alpha, i, &s_i_0, t_i_0, &public.cs[i]);
            assert!(pi_0 == pi_1, "proofs must be equal at each level");
        }
    }
}
