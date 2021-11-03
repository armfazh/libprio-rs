// SPDX-License-Identifier: MPL-2.0

//! **(NOTE: This module is experimental. Applications should not use it yet.)** This modulde
//! implements the `prio3` [VDAF]. The construction is based on a transform of a Fully Linear Proof
//! (FLP) system (i.e., a concrete [`Value`](`crate::pcp::Value`) into a zero-knowledge proof
//! system on distributed data as described in [[BBCG+19], Section 6].
//!
//! TODO Align this module with [`crate::vdaf::Vdaf`].
//!
//! [BBCG+19]: https://ia.cr/2019/188
//! [BBCG+21]: https://ia.cr/2021/017
//! [VDAF]: https://datatracker.ietf.org/doc/draft-patton-cfrg-vdaf/

use crate::field::FieldElement;
use crate::pcp::{decide, prove, query, Proof, Value, ValueParam, Verifier};
use crate::prng::Prng;
use crate::vdaf::suite::{Key, KeyDeriver, KeyStream, Suite};
use crate::vdaf::{
    Aggregator, Client, Collector, PrepareStep, PrepareTransition, Share, Vdaf, VdafError,
};
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::iter::IntoIterator;
use std::marker::PhantomData;

struct Prio3<'a, V, M, A>
where
    V: Value<'a>,
{
    suite: Suite,
    param: V::Param,
    num_aggregators: u8,
    phantom: PhantomData<&'a (V, M, A)>,
}

impl<'a, V, M, A> Vdaf for Prio3<'a, V, M, A>
where
    V: Value<'a>,
{
    type Measurement = M;
    type AggregateResult = A;
    type AggregationParam = ();
    type PublicParam = ();
    type VerifyParam = Prio3VerifyParam;
    type InputShare = Prio3InputShare;
    type OutputShare = Vec<V::Field>;
    type AggregateShare = Vec<V::Field>;

    fn setup(&self) -> Result<((), Vec<Prio3VerifyParam>), VdafError> {
        panic!("XXX");
    }

    fn num_aggregators(&self) -> usize {
        self.num_aggregators as usize
    }
}

pub struct Prio3VerifyParam {}

pub struct Prio3InputShare {}

impl<'a, V, M, A> Client for Prio3<'a, V, M, A>
where
    V: Value<'a> + for<'b> TryFrom<(&'b M, &'b V::Param), Error = VdafError>,
{
    fn shard(&self, public_param: &(), measurement: &M) -> Result<Vec<Prio3InputShare>, VdafError> {
        let input = V::try_from((measurement, &self.param))?;
        panic!("XXX");
    }
}

pub struct Prio3PrepareStep {}

impl<'a, V, M, A> Aggregator for Prio3<'a, V, M, A>
where
    V: Value<'a>,
{
    /// Initial state of the Aggregator during the Prepare process.
    type PrepareInit = Prio3PrepareStep;

    /// Begins the Prepare process with the other Aggregators. The result of this process is
    /// the Aggregator's output share.
    fn prepare(
        &self,
        verify_param: &Prio3VerifyParam,
        _agg_param: &(),
        nonce: &[u8],
        input_share: &Prio3InputShare,
    ) -> Result<Prio3PrepareStep, VdafError> {
        panic!("XXX");
    }

    /// Aggregates a sequence of output shares into an aggregate share.
    fn aggregate<It: IntoIterator<Item = Vec<V::Field>>>(
        &self,
        _agg_param: &(),
        output_shares: It,
    ) -> Result<Vec<V::Field>, VdafError> {
        panic!("XXX");
    }
}

pub struct Prio3Verify {}

impl<F: FieldElement> PrepareStep<Vec<F>> for Prio3PrepareStep {
    type Input = Prio3Verify;
    type Output = Prio3Verify;

    fn step<M: IntoIterator<Item = Prio3Verify>>(
        mut self,
        inputs: M,
    ) -> PrepareTransition<Self, Vec<F>> {
        panic!("XXX");
    }
}

impl<'a, V, M, A> Collector for Prio3<'a, V, M, A>
where
    V: Value<'a>,
{
    /// Combines aggregate shares into the aggregate result.
    fn unshard<It: IntoIterator<Item = Vec<V::Field>>>(
        &self,
        _agg_param: &(),
        agg_shares: It,
    ) -> Result<Self::AggregateResult, VdafError> {
        panic!("XXX");
    }
}

/// The message sent by the client to each aggregator. This includes the client's input share and
/// the initial message of the input-validation protocol.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputShareMessage<F> {
    /// The input share.
    pub input_share: Share<F>,

    /// The proof share.
    pub proof_share: Share<F>,

    /// The sum of the joint randomness seed shares sent to the other aggregators.
    //
    // TODO(cjpatton) If `input.joint_rand_len() == 0`, then we don't need to bother with the
    // joint randomness seed at all and make this optional.
    pub joint_rand_seed_hint: Key,

    /// The blinding factor, used to derive the aggregator's joint randomness seed share.
    pub blind: Key,
}

#[derive(Clone)]
struct HelperShare {
    input_share: Key,
    proof_share: Key,
    joint_rand_seed_hint: Key,
    blind: Key,
}

impl HelperShare {
    fn new(suite: Suite) -> Result<Self, VdafError> {
        Ok(HelperShare {
            input_share: Key::generate(suite)?,
            proof_share: Key::generate(suite)?,
            joint_rand_seed_hint: Key::uninitialized(suite),
            blind: Key::generate(suite)?,
        })
    }
}

/// The input-distribution algorithm of the VDAF. Run by the client, this generates the sequence of
/// [`InputShareMessage`] messages to send to the aggregators. Note that this particular VDAF does
/// not have a public parameter.
///
/// # Parameters
///
/// * `suite` is the cipher suite used for key derivation.
/// * `input` is the input to be secret shared.
/// * `num_shares` is the number of input shares (i.e., aggregators) to generate.
pub fn prio3_input<'a, V: Value<'a>>(
    suite: Suite,
    input: &V,
    num_shares: u8,
) -> Result<Vec<InputShareMessage<V::Field>>, VdafError> {
    check_num_shares("prio3_input", num_shares)?;

    let input_len = input.as_slice().len();
    let num_shares = num_shares as usize;
    let param = input.param();

    // Generate the input shares and compute the joint randomness.
    let mut helper_shares = vec![HelperShare::new(suite)?; num_shares - 1];
    let mut leader_input_share = input.as_slice().to_vec();
    let mut joint_rand_seed = Key::uninitialized(suite);
    let mut aggregator_id = 1; // ID of the first helper
    for helper in helper_shares.iter_mut() {
        let mut deriver = KeyDeriver::from_key(&helper.blind);
        deriver.update(&[aggregator_id]);
        let prng: Prng<V::Field> = Prng::from_key_stream(KeyStream::from_key(&helper.input_share));
        for (x, y) in leader_input_share.iter_mut().zip(prng).take(input_len) {
            *x -= y;
            deriver.update(&y.into());
        }

        helper.joint_rand_seed_hint = deriver.finish();
        for (x, y) in joint_rand_seed
            .as_mut_slice()
            .iter_mut()
            .zip(helper.joint_rand_seed_hint.as_slice().iter())
        {
            *x ^= y;
        }

        aggregator_id += 1; // ID of the next helper
    }

    let leader_blind = Key::generate(suite)?;

    let mut deriver = KeyDeriver::from_key(&leader_blind);
    deriver.update(&[0]); // ID of the leader
    for x in leader_input_share.iter() {
        deriver.update(&(*x).into());
    }

    let mut leader_joint_rand_seed_hint = deriver.finish();
    for (x, y) in joint_rand_seed
        .as_mut_slice()
        .iter_mut()
        .zip(leader_joint_rand_seed_hint.as_slice().iter())
    {
        *x ^= y;
    }

    // Run the proof-generation algorithm.
    let prng: Prng<V::Field> = Prng::from_key_stream(KeyStream::from_key(&joint_rand_seed));
    let joint_rand: Vec<V::Field> = prng.take(param.joint_rand_len()).collect();
    let prng: Prng<V::Field> = Prng::generate(suite)?;
    let prove_rand: Vec<V::Field> = prng.take(param.prove_rand_len()).collect();
    let proof = prove(input, &prove_rand, &joint_rand)?;

    // Generate the proof shares and finalize the joint randomness seed hints.
    let proof_len = proof.as_slice().len();
    let mut leader_proof_share = proof.data;
    for helper in helper_shares.iter_mut() {
        let prng: Prng<V::Field> = Prng::from_key_stream(KeyStream::from_key(&helper.proof_share));
        for (x, y) in leader_proof_share.iter_mut().zip(prng).take(proof_len) {
            *x -= y;
        }

        for (x, y) in helper
            .joint_rand_seed_hint
            .as_mut_slice()
            .iter_mut()
            .zip(joint_rand_seed.as_slice().iter())
        {
            *x ^= y;
        }
    }

    for (x, y) in leader_joint_rand_seed_hint
        .as_mut_slice()
        .iter_mut()
        .zip(joint_rand_seed.as_slice().iter())
    {
        *x ^= y;
    }

    // Prepare the output messages.
    let mut out = Vec::with_capacity(num_shares);
    out.push(InputShareMessage {
        input_share: Share::Leader(leader_input_share),
        proof_share: Share::Leader(leader_proof_share),
        joint_rand_seed_hint: leader_joint_rand_seed_hint,
        blind: leader_blind,
    });

    for helper in helper_shares.into_iter() {
        out.push(InputShareMessage {
            input_share: Share::Helper(helper.input_share),
            proof_share: Share::Helper(helper.proof_share),
            joint_rand_seed_hint: helper.joint_rand_seed_hint,
            blind: helper.blind,
        });
    }

    Ok(out)
}

/// The verification parameter used by each aggregator to evaluate the VDAF.
#[derive(Clone, Debug)]
pub struct VerifyParam<'a, V: Value<'a>> {
    /// Input parameter, needed to reconstruct a [`Value`] from a vector of field elements.
    pub value_param: V::Param,

    /// Key used to derive the query randomness from the nonce.
    pub query_rand_init: Key,

    /// The identity of the aggregator.
    pub aggregator_id: u8,

    /// The number of aggregators.
    pub num_shares: u8,
}

/// The setup algorithm of the VDAF that generates the verification parameter of each aggregator.
/// Note that this VDAF does not involve a public parameter.
#[cfg(test)]
fn prio3_setup<'a, V: Value<'a>>(
    suite: Suite,
    value_param: &V::Param,
    num_shares: u8,
) -> Result<Vec<VerifyParam<'a, V>>, VdafError> {
    check_num_shares("prio3_setup", num_shares)?;

    let query_rand_init = Key::generate(suite)?;
    Ok((0..num_shares)
        .map(|aggregator_id| VerifyParam {
            value_param: value_param.clone(),
            query_rand_init: query_rand_init.clone(),
            aggregator_id,
            num_shares,
        })
        .collect())
}

/// The message sent by an aggregator to every other aggregator. This is the final message of the
/// input-validation protocol.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyMessage<F> {
    /// The aggregator's share of the verifier message.
    pub verifier_share: Verifier<F>,

    /// The aggregator's share of the joint randomness, derived from its input share.
    //
    // TODO(cjpatton) If `joint_rand_len == 0`, then we don't need to bother with the
    // joint randomness seed at all and make this optional.
    pub joint_rand_seed_share: Key,
}

/// The state of each aggregator.
#[derive(Clone, Debug)]
pub struct VerifyState<'a, V: Value<'a>> {
    /// The input share.
    input_share: V,

    /// The joint randomness seed indicated by the client. The aggregators check that this
    /// indication matches the actual joint randomness seed.
    //
    // TODO(cjpatton) If `joint_rand_len == 0`, then we don't need to bother with the
    // joint randomness seed at all and make this optional.
    joint_rand_seed: Key,

    phantom: PhantomData<&'a V>,
}

/// The verify-start algorithm of the VDAF. Run by each aggregator, this consumes the
/// [`InputShareMessage`] message sent from the client and produces the [`VerifyMessage`] message
/// that will be broadcast to the other aggregators. This VDAF does not involve an aggregation
/// parameter.
//
// TODO(cjpatton) Check for ciphersuite mismatch between `verify_param` and `msg`.
pub fn prio3_start<'a, V: Value<'a>>(
    verify_param: &'a VerifyParam<'a, V>,
    nonce: &[u8],
    msg: InputShareMessage<V::Field>,
) -> Result<(VerifyState<'a, V>, VerifyMessage<V::Field>), VdafError> {
    let param = &(verify_param.value_param);

    let mut deriver = KeyDeriver::from_key(&verify_param.query_rand_init);
    deriver.update(&[255]);
    deriver.update(nonce);
    let query_rand_seed = deriver.finish();

    let input_share_data: Vec<V::Field> = msg.input_share.into_vec(param.input_len())?;
    let input_share = V::new_share(input_share_data, param, verify_param.num_shares as usize)?;

    let proof_share_data: Vec<V::Field> = msg.proof_share.into_vec(param.proof_len())?;
    let proof_share = Proof::from(proof_share_data);

    // Compute the joint randomness.
    let mut deriver = KeyDeriver::from_key(&msg.blind);
    deriver.update(&[verify_param.aggregator_id]);
    for x in input_share.as_slice() {
        deriver.update(&(*x).into());
    }
    let joint_rand_seed_share = deriver.finish();

    let mut joint_rand_seed = Key::uninitialized(query_rand_seed.suite());
    for (j, x) in joint_rand_seed.as_mut_slice().iter_mut().enumerate() {
        *x = msg.joint_rand_seed_hint.as_slice()[j] ^ joint_rand_seed_share.as_slice()[j];
    }

    let prng: Prng<V::Field> = Prng::from_key_stream(KeyStream::from_key(&joint_rand_seed));
    let joint_rand: Vec<V::Field> = prng.take(param.joint_rand_len()).collect();

    // Compute the query randomness.
    let prng: Prng<V::Field> = Prng::from_key_stream(KeyStream::from_key(&query_rand_seed));
    let query_rand: Vec<V::Field> = prng.take(param.query_rand_len()).collect();

    // Run the query-generation algorithm.
    let verifier_share = query(&input_share, &proof_share, &query_rand, &joint_rand)?;

    // Prepare the output state and message.
    let state = VerifyState {
        input_share,
        joint_rand_seed,
        phantom: PhantomData,
    };

    let out = VerifyMessage {
        verifier_share,
        joint_rand_seed_share,
    };

    Ok((state, out))
}

/// The verify-finish algorithm of the VDAF. Run by each aggregator, this consumes the
/// [`VerifyMessage`] messages broadcast by all of the aggregators and produces the aggregator's
/// input share.
//
// TODO(cjpatton) Check for ciphersuite mismatch between `state` and `msgs` and among `msgs`.
pub fn prio3_finish<'a, M, V>(mut state: VerifyState<'a, V>, msgs: M) -> Result<V, VdafError>
where
    V: Value<'a>,
    M: IntoIterator<Item = VerifyMessage<V::Field>>,
{
    let mut msgs = msgs.into_iter().peekable();

    let verifier_length = match msgs.peek() {
        Some(message) => message.verifier_share.as_slice().len(),
        None => {
            return Err(VdafError::Uncategorized(
                "prio3_finish(): expected at least one inbound messages; got none".to_string(),
            ));
        }
    };

    // Combine the verifier messages.
    let mut verifier_data = vec![V::Field::zero(); verifier_length];
    for msg in msgs {
        if msg.verifier_share.as_slice().len() != verifier_data.len() {
            return Err(VdafError::Uncategorized(format!(
                "prio3_finish(): expected verifier share of length {}; got {}",
                verifier_data.len(),
                msg.verifier_share.as_slice().len(),
            )));
        }

        for (x, y) in verifier_data.iter_mut().zip(msg.verifier_share.as_slice()) {
            *x += *y;
        }

        for (x, y) in state
            .joint_rand_seed
            .as_mut_slice()
            .iter_mut()
            .zip(msg.joint_rand_seed_share.as_slice())
        {
            *x ^= y;
        }
    }

    // Check that the joint randomness was correct.
    if state.joint_rand_seed != Key::uninitialized(state.joint_rand_seed.suite()) {
        return Err(VdafError::Uncategorized(
            "joint randomness mismatch".to_string(),
        ));
    }

    // Check the proof.
    let verifier = Verifier::from(verifier_data);
    let result = decide(state.input_share.param(), &verifier)?;
    if !result {
        return Err(VdafError::Uncategorized("proof check failed".to_string()));
    }

    Ok(state.input_share)
}

fn check_num_shares(func_name: &str, num_shares: u8) -> Result<(), VdafError> {
    if num_shares == 0 {
        return Err(VdafError::Uncategorized(format!(
            "{}(): at least one share is required; got {}",
            func_name, num_shares
        )));
    } else if num_shares > 254 {
        return Err(VdafError::Uncategorized(format!(
            "{}(): share count must not exceed 254; got {}",
            func_name, num_shares
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::Field64;
    use crate::pcp::types::{Count, CountParam};

    #[test]
    fn test_prio3() {
        let suite = Suite::Blake3;
        const NUM_SHARES: usize = 23;

        let param = CountParam::new();
        let input: Count<Field64> = Count::new(1, &param).unwrap();
        let nonce = b"This is a good nonce.";

        // Client runs the input and proof distribution algorithms.
        let input_shares = prio3_input(suite, &input, NUM_SHARES as u8).unwrap();

        // Aggregators agree on seed used to generate per-report query randomness.
        let verify_params = prio3_setup(suite, &param, NUM_SHARES as u8).unwrap();

        // Aggregators receive their proof shares and broadcast their verifier messages.
        let (states, verifiers): (
            Vec<VerifyState<Count<Field64>>>,
            Vec<VerifyMessage<Field64>>,
        ) = verify_params
            .iter()
            .zip(input_shares.into_iter())
            .map(|(verify_param, input_share)| {
                prio3_start(verify_param, &nonce[..], input_share).unwrap()
            })
            .unzip();

        // Aggregators decide whether the input is valid based on the verifier messages.
        let mut output = vec![Field64::zero(); input.as_slice().len()];
        for state in states {
            let output_share = prio3_finish(state, verifiers.clone()).unwrap();
            for (x, y) in output.iter_mut().zip(output_share.as_slice()) {
                *x += *y;
            }
        }

        assert_eq!(input.as_slice(), output.as_slice());
    }
}
