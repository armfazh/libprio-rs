// Primitives

// #[rustversion::stable]
// #[inline(always)]
// pub fn fp128_addcarryx_u64(
//     out1: &mut u64,
//     out2: &mut Fp128U1,
//     arg1: Fp128U1,
//     arg2: u64,
//     arg3: u64,
// ) {
//     let x1: u128 = (((arg1 as u128) + (arg2 as u128)) + (arg3 as u128));
//     let x2: u64 = ((x1 & (0xffffffffffffffff_u128)) as u64);
//     let x3: Fp128U1 = ((x1 >> 64) as Fp128U1);
//     *out1 = x2;
//     *out2 = x3;
// }

// #[rustversion::stable]
// #[inline(always)]
// pub fn fp128_subborrowx_u64(
//     out1: &mut u64,
//     out2: &mut Fp128U1,
//     arg1: Fp128U1,
//     arg2: u64,
//     arg3: u64,
// ) {
//     let x1: i128 = (((arg2 as i128) - (arg1 as i128)) - (arg3 as i128));
//     let x2: Fp128I1 = ((x1 >> 64) as Fp128I1);
//     let x3: u64 = ((x1 & (0xffffffffffffffff as i128)) as u64);
//     *out1 = x3;
//     *out2 = (((0x0 as Fp128I2) - (x2 as Fp128I2)) as Fp128U1);
// }

// #[rustversion::stable]
// #[inline(always)]
// pub fn fp128_mulx_u64(out1: &mut u64, out2: &mut u64, arg1: u64, arg2: u64) {
//     let x1: u128 = ((arg1 as u128) * (arg2 as u128));
//     let x2: u64 = ((x1 & (0xffffffffffffffff as u128)) as u64);
//     let x3: u64 = ((x1 >> 64) as u64);
//     *out1 = x2;
//     *out2 = x3;
// }

// #[rustversion::nightly]
// macro_rules! fp128_addcarryx_u64 {
//     (&mut $u0:ident,&mut $u1:ident,$u2:expr,$u3:expr,$u4:expr) => {{
//         let (_0, _1) = $u3.carrying_add($u4, $u2 != 0);
//         (*$u0, *$u1) = (_0, _1 as u8);
//     }};
// }

// #[rustversion::nightly]
// macro_rules! fp128_subborrowx_u64 {
//     (&mut $u0:ident,&mut $u1:ident,$u2:expr,$u3:expr,$u4:expr) => {{
//         let (_0, _1) = $u3.borrowing_sub($u4, $u2 != 0);
//         (*$u0, *$u1) = (_0, _1 as u8);
//     }};
// }

// #[rustversion::nightly]
// macro_rules! fp128_mulx_u64 {
//     (&mut $u0:ident,&mut $u1:ident,$u2:expr,$u3:expr) => {{
//         let (_0, _1) = $u2.widening_mul($u3);
//         *$u0 = _0;
//         *$u1 = _1;
//     }};
// }

// #[rustversion::nightly]
// #[inline(always)]
// pub fn fp128_addcarryx_u64(
//     out1: &mut u64,
//     out2: &mut Fp128U1,
//     arg1: Fp128U1,
//     arg2: u64,
//     arg3: u64,
// ) {
//     (out1, out2) = arg2.carrying_add(arg3, arg1 != 0)
// }

// #[rustversion::nightly]
// #[inline(always)]
// pub fn fp128_subborrowx_u64(
//     out1: &mut u64,
//     out2: &mut Fp128U1,
//     arg1: Fp128U1,
//     arg2: u64,
//     arg3: u64,
// ) {
//     (out1, out2) = arg2.borrowing_sub(arg3, arg1 != 0)
// }

// #[rustversion::nightly]
// #[inline(always)]
// pub fn fp128_mulx_u64(out1: &mut u64, out2: &mut u64, arg1: u64, arg2: u64) {
//     (out1, out2) = arg1.widening_mul(arg2)
// }

// #[inline(always)]
// pub fn fp128_cmovznz_u64(out1: &mut u64, arg1: Fp128U1, arg2: u64, arg3: u64) {
//     let x1: Fp128U1 = (!(!arg1));
//     let x2: u64 = ((((((0x0 as Fp128I2) - (x1 as Fp128I2)) as Fp128I1) as i128)
//         & (0xffffffffffffffff as i128)) as u64);
//     let x3: u64 = ((x2 & arg3) | ((!x2) & arg2));
//     *out1 = x3;
// }
