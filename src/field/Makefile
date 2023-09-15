FIAT_PATH=fiat-crypto
FIAT_BINARY=word_by_word_montgomery
OPERS=add sub mul opp from_montgomery to_montgomery to_bytes from_bytes one

#Flags to check with tests:
# --no-primitives
# --use-value-barrier
# --relax-primitive-carry-to-bitwidth


all: f128

f32: PRIME=4293918721
f32: fp32.rs

f64: PRIME=18446744069414584321
f64: fp64.rs

f128: PRIME="(2^62-2^3+1)*2^66+1" # PRIME=340282366920938462946865773367900766209
f128: fiat_crypto_fp128.rs

fp128.c:
	${FIAT_PATH}/${FIAT_BINARY} --output $@ \
		--lang C \
		--no-prefix-fiat --inline-internal \
		Fp128 64 ${PRIME} \
		${OPERS}

fiat_crypto_fp%.rs:
	${FIAT_PATH}/${FIAT_BINARY}           \
		--output $@                       \
		--lang Rust                       \
		--inline                          \
		--inline-internal                 \
		--no-prefix-fiat                  \
		--public-function-case snake_case \
		--private-function-case camelCase \
		--public-type-case UpperCamelCase \
		fp$* 64 ${PRIME} ${OPERS}

clean:
	rm -f fiat_crypto_fp128.rs
