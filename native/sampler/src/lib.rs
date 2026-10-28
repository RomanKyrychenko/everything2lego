//! The statistics sampler's inner loop (everything2lego/sampler.py), native.
//!
//! Python prepares everything that is data — the target's cells, the
//! vocabulary's parts in their yaws, the neighbourhood statistics as dense
//! token tables — and calls `e2l_sample` once.  This file holds the state
//! (which part owns each cell, each cell's shape, the running energy
//! terms) and runs the fill, the warm-up and the annealing walk exactly as
//! sampler.py's State, Filler, retile, merge and sample do.  The two must
//! stay in step: a change to the energy or the moves goes in both.
//!
//! Grid: (y plates up, x studs, z studs), cell index (y * X + x) * Z + z.
//! A cell's shape is a 64-bit mask of 4x4x4 samples.

use std::panic;
use std::slice;

// ---------------------------------------------------------------------------
// the problem, as Python lays it out
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct Problem {
    // target
    pub h: i32,
    pub x: i32,
    pub z: i32,
    pub tbits: *const u64,   // H*X*Z: each cell's shape
    pub want: *const u8,     // H*X*Z
    pub visible: *const u8,  // H*X*Z
    pub tcolour: *const i32, // H*X*Z palette index, -1 unknown
    pub tlab: *const f32,    // H*X*Z*3, valid where tcolour >= 0
    // palette
    pub n_col: i32,
    pub pal_lab: *const f32, // n_col*3
    // statistics over the vocabulary's tokens; direction 0 up, 1 down, 2 level
    pub n_tok: i32,
    pub nll: *const f64,  // n_tok*n_tok*3: -log P(B, where | A)
    pub logp: *const f64, // n_tok*n_tok*3: log p_ref(A, B, where)
    // parts: the colour each palette colour becomes on it (nearest it is made in)
    pub n_part: i32,
    pub snap: *const i32, // n_part*n_col
    // options: a part in one yaw
    pub n_opt: i32,
    pub opt_part: *const i32,
    pub opt_token: *const i32,
    pub opt_size: *const i32,  // n_opt*3: h, W, D
    pub opt_first: *const i32, // n_opt*3: its first occupied cell
    pub opt_logprior: *const f64,
    pub opt_box: *const i32, // -1, or 0 brick / 1 plate / 2 tile for a plain full box
    pub cell_start: *const i32, // n_opt+1
    pub cell_yxz: *const i32,   // n_cells*3
    pub cell_bits: *const u64,
    pub cell_o: *const f64,
    pub top_start: *const i32, // n_opt+1
    pub top_xz: *const i32,
    pub bot_start: *const i32,
    pub bot_xz: *const i32,
    pub side_start: *const i32,
    pub side_v: *const i32, // n_sides*6: hx, hy, hz (half studs / plates from the box corner), dx, dz, kind
    pub halo_start: *const i32, // n_opt+1
    pub halo_yxz: *const i32,   // n_halo*3: the cells next to its cells (26-neighbourhood), not its own
    pub n_conn: i32,
    pub fits: *const u8, // n_conn*n_conn: which side connector kinds join
    // weights and schedule
    pub w_cov: f64,
    pub w_col: f64,
    pub w_loc: f64,
    pub w_glob: f64,
    pub w_sup: f64,
    pub w_side: f64,
    pub n_ref: f64, // pairs a real set of the target's volume has
    pub w_dens: f64,
    pub dens_sd: f64, // spread of log(pairs per cell) over the reference models
    pub p_ref: f64,   // parts a real set of the target's volume has
    pub w_cnt: f64,
    pub cnt_sd: f64,
    pub w_sil: f64,
    pub style: f64,
    pub w_glob2: f64,
    pub logp2: *const f64, // n_tok*n_tok*3: log p_ref of path patterns (a - b - c as (a, c, where))
    pub middle_w: *const f64, // n_tok: a path's weight through a part of this token
    // 4-walk patterns a - b - c - d, keyed by (a_tok, d_tok, direction, mid_bucket)
    // with mid_bucket an 8-way hash of the sorted pair of middle kinds.
    pub w_glob3: f64,
    pub logp3: *const f64,      // n_tok*n_tok*3*8 or null if disabled
    pub mid_bucket: *const i32, // n_tok*n_tok or null: bucket for the middle pair (b, c)
    pub max_deg_paths4: i32,    // degree cap: parts with more neighbours skip paths4 (0 = no cap)
    pub paths4_sample_n: i32,   // deterministic subsample: keep walks where hash(a,b,c,d) % N == 0 (0/1 = all)
    pub w_entropy_glob3: f64,   // anti mode-collapse: energy -= w_entropy_glob3 * H(kl3) (nats)
    pub w_revkl_glob3: f64,     // coverage term: energy -= w_revkl_glob3 * (s_plogc / n_ours)
    pub pref3: *const f64,      // n_tok*n_tok*3*8 or null: p_ref values parallel to logp3
    // each option's outline in the three axis views, sample pixels box-local:
    // along x (sy, sz), along y (sx, sz), along z (sy, sx)
    pub pixx_start: *const i32,
    pub pixx: *const i32,
    pub pixy_start: *const i32,
    pub pixy: *const i32,
    pub pixz_start: *const i32,
    pub pixz: *const i32,
    // the target's outline in those views: (H*4, Z*4), (X*4, Z*4), (H*4, X*4)
    pub tsx: *const u8,
    pub tsy: *const u8,
    pub tsz: *const u8,
    // the proposal network (everything2lego/proposals.py); use_net 0: the pair statistics propose
    pub use_net: i32,
    pub net_in: i32,
    pub net_h: i32,
    pub net_c: i32,
    pub net_kinds: i32,
    pub net_objects: i32,
    pub net_object: i32,
    pub w1: *const f32, // net_in*net_h
    pub b1: *const f32,
    pub w2: *const f32, // net_h*net_c
    pub b2: *const f32,
    pub mu: *const f32,
    pub sd: *const f32,
    pub opt_out: *const i32,   // n_opt: the network output of its token
    pub opt_lpart: *const f64, // n_opt: log P(part | token)
    pub opt_kind: *const i32,  // n_opt: its kind index, for the context
    pub region_object: *const i32, // H*X*Z: per-cell object index, -1 = use net.object
    pub opt_variants: *const i32,  // n_opt: swap-group id (same box kind + size); -1 none
    pub net_static: i32,       // the wider view's features per cell (proposals.static_context)
    pub static_: *const f32,   // H*X*Z*net_static
    // hinge pairs (everything2lego/hinges.py; tools/partsdb/hinges.py): each entry
    // is (opt_a, opt_b, dy, dx, dz) in sampler grid units, with opt_a/b indexing
    // into the opts array.  See hinge_swap() for how the pose is applied.
    pub n_hinge: i32,
    pub hinge_data: *const i32,   // n_hinge*5: (opt_a, opt_b, dy, dx, dz)
    // multi-scale warm-start (sampler._place_init / tools/multi_scale.py):
    // placements from a coarse pass that pre-populate the state before filler.
    pub n_init: i32,
    pub init_opt: *const i32,     // n_init
    pub init_colour: *const i32,  // n_init
    pub init_y: *const i32,       // n_init
    pub init_x: *const i32,       // n_init
    pub init_z: *const i32,       // n_init
    // per-cell architectural region label (classify_regions.LABEL); -1 absent.
    // masonry_shift keys on label==1 (wall).
    pub region_label: *const i32, // H*X*Z or null
    pub t_start_share: f64,
    pub t_final_share: f64,
    pub iters: i64,
    pub warmup: i64,
    pub seed: u64,
    pub top_k: i32,
    pub log_every: i64,
    // outputs
    pub out_cap: i32,
    pub out_n: *mut i32,
    pub out_opt: *mut i32,
    pub out_colour: *mut i32,
    pub out_y: *mut i32,
    pub out_x: *mut i32,
    pub out_z: *mut i32,
    pub info: *mut f64, // INFO_LEN values, see `run`
    pub hist_cap: i32,
    pub hist_n: *mut i32,
    pub hist: *mut f64,
    // part-size statistics: energy += w_size * n_parts * KL(build's size-class mix || real sets'),
    // classes by footprint and height (sizes.py); null / 0 = off
    pub w_size: f64,
    pub n_size_class: i32,
    pub opt_class: *const i32, // n_opt: size class of each option
    pub size_pref: *const f64, // n_size_class: the classes' share in real sets
    // sideways building (snot_options.pair_rows): rows of 7 ints (plain brick option, its side-stud anchor option,
    // a sideways plate / tile option, dy, dx, dz of the tile's box from the brick's, which anchor stud it hangs on)
    pub n_snot_rows: i32,
    pub snot_rows: *const i32,
    // sideways share: energy += w_sideways * n_parts * KL(Bernoulli(build's sideways share) || p_sideways); 0 = off
    pub w_sideways: f64,
    pub p_sideways: f64,
    pub opt_sideways: *const u8, // n_opt: 1 = a sideways option
    // surface fit (surface_fit.py): per cell and sample (bit order of the masks), the cost of LEGO there when the
    // target is empty (grows with the distance to the target's surface) and the reward for covering it when the
    // target is full (1 near the outer surface, ~0 deep inside); null = the plain count of wrong samples
    pub w_out: *const f32, // h*x*z*64
    pub w_in: *const f32,  // h*x*z*64
}

const INFO_LEN: usize = 20;
const PATHS4_BUCKETS: usize = 8;
const NSAMP: f64 = 64.0;
const SUB: i32 = 4;

// ---------------------------------------------------------------------------
// random numbers: xoshiro256**, seeded by splitmix64
// ---------------------------------------------------------------------------

struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Rng {
        let mut z = seed;
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            *v = x ^ (x >> 31);
        }
        Rng { s }
    }
    fn next(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }
    fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.f64() * n as f64) as usize % n.max(1)
    }
}

// ---------------------------------------------------------------------------
// data
// ---------------------------------------------------------------------------

struct Opt {
    part: usize,
    token: usize,
    h: i32,
    w: i32,
    d: i32,
    first: (i32, i32, i32),
    logprior: f64,
    boxk: i32,
    cells: Vec<(i32, i32, i32, u64, f64)>,
    top: Vec<(i32, i32)>,
    bottom: Vec<(i32, i32)>,
    sides: Vec<(i32, i32, i32, i32, i32, i32)>,
    halo: Vec<(i32, i32, i32)>,
    pix: [Vec<(i32, i32)>; 3],
    top_mask: Vec<bool>, // W*D
    bot_mask: Vec<bool>,
}

impl Opt {
    fn has_top(&self, dx: i32, dz: i32) -> bool {
        dx >= 0 && dz >= 0 && dx < self.w && dz < self.d && self.top_mask[(dx * self.d + dz) as usize]
    }
    fn has_bottom(&self, dx: i32, dz: i32) -> bool {
        dx >= 0 && dz >= 0 && dx < self.w && dz < self.d && self.bot_mask[(dx * self.d + dz) as usize]
    }
}

struct Data {
    h: i32,
    x: i32,
    z: i32,
    tbits: Vec<u64>,
    fill: Vec<f64>,
    want_cells: Vec<usize>,
    want: Vec<bool>,
    visible: Vec<bool>,
    tcolour: Vec<i32>,
    tlab: Vec<f32>,
    n_col: usize,
    pal_lab: Vec<f32>,
    n_tok: usize,
    nll: Vec<f64>,
    logp: Vec<f64>,
    snap: Vec<i32>,
    opts: Vec<Opt>,
    boxes: std::collections::HashMap<(i32, i32, i32, i32), usize>,
    variants: std::collections::HashMap<i32, Vec<usize>>,
    opt_variants: Vec<i32>,
    w_cov: f64,
    w_col: f64,
    w_loc: f64,
    w_glob: f64,
    w_sup: f64,
    w_side: f64,
    n_ref: f64,
    w_dens: f64,
    dens_sd: f64,
    p_ref: f64,
    w_cnt: f64,
    cnt_sd: f64,
    w_sil: f64,
    style: f64,
    w_glob2: f64,
    logp2: Vec<f64>,
    middle_w: Vec<f64>,
    w_glob3: f64,
    logp3: Vec<f64>,       // n_tok*n_tok*3*8 or empty
    mid_bucket: Vec<i32>,  // n_tok*n_tok or empty
    max_deg_paths4: i32,
    paths4_sample_n: u32,
    w_entropy_glob3: f64,
    w_revkl_glob3: f64,
    pref3: Vec<f64>,
    tsil: [Vec<bool>; 3],
    scols: [i32; 3], // columns of each view (rows are implied)
    n_conn: usize,
    fits: Vec<u8>,
    net: Option<Net>,
    top_k: usize,
    hinge_a: Vec<usize>,
    hinge_b: Vec<usize>,
    hinge_pose: Vec<[i32; 3]>, // (dy, dx, dz)
    hinge_by_a: std::collections::HashMap<usize, Vec<usize>>, // opt_a -> indices into hinge_a/b/pose
    region_label: Vec<i32>,    // H*X*Z or empty
    w_size: f64,
    opt_class: Vec<usize>,     // n_opt (empty when off)
    size_lpref: Vec<f64>,      // ln of the real sets' class shares
    snot_rows: Vec<(usize, usize, i32, i32, i32, i32)>, // (anchor, tile, dy, dx, dz, stud)
    snot_by_opt: Vec<Vec<usize>>,                         // brick option -> its rows (empty when off)
    w_sideways: f64,
    p_sideways: f64,
    opt_sideways: Vec<bool>,                              // n_opt (empty when off)
    w_out: Vec<f32>,                                      // n*64 (empty: plain sample count)
    w_in: Vec<f32>,
    fid0: f64,                                            // the shape term of an empty build
}

unsafe fn v<T: Copy>(p: *const T, n: usize) -> Vec<T> {
    if n == 0 {
        return Vec::new();
    }
    slice::from_raw_parts(p, n).to_vec()
}

impl Data {
    unsafe fn from(p: &Problem) -> Data {
        let n = (p.h * p.x * p.z) as usize;
        let tbits = v(p.tbits, n);
        let fill = tbits.iter().map(|b| b.count_ones() as f64 / NSAMP).collect();
        let want: Vec<bool> = v(p.want, n).iter().map(|&b| b != 0).collect();
        let want_cells = (0..n).filter(|&i| want[i]).collect();
        let n_opt = p.n_opt as usize;
        let size = v(p.opt_size, n_opt * 3);
        let first = v(p.opt_first, n_opt * 3);
        let part = v(p.opt_part, n_opt);
        let token = v(p.opt_token, n_opt);
        let lp = v(p.opt_logprior, n_opt);
        let bx = v(p.opt_box, n_opt);
        let cs = v(p.cell_start, n_opt + 1);
        let n_cells = cs[n_opt] as usize;
        let cyxz = v(p.cell_yxz, n_cells * 3);
        let cbits = v(p.cell_bits, n_cells);
        let co = v(p.cell_o, n_cells);
        let ts = v(p.top_start, n_opt + 1);
        let txz = v(p.top_xz, ts[n_opt] as usize * 2);
        let bs = v(p.bot_start, n_opt + 1);
        let bxz = v(p.bot_xz, bs[n_opt] as usize * 2);
        let ss = v(p.side_start, n_opt + 1);
        let sv = v(p.side_v, ss[n_opt] as usize * 6);
        let hs = v(p.halo_start, n_opt + 1);
        let hv = v(p.halo_yxz, hs[n_opt] as usize * 3);
        let pstart = [v(p.pixx_start, n_opt + 1), v(p.pixy_start, n_opt + 1), v(p.pixz_start, n_opt + 1)];
        let pvals = [
            v(p.pixx, pstart[0][n_opt] as usize * 2),
            v(p.pixy, pstart[1][n_opt] as usize * 2),
            v(p.pixz, pstart[2][n_opt] as usize * 2),
        ];
        let mut opts = Vec::with_capacity(n_opt);
        let mut boxes = std::collections::HashMap::new();
        for k in 0..n_opt {
            let (h, w, d) = (size[3 * k], size[3 * k + 1], size[3 * k + 2]);
            let cells = (cs[k] as usize..cs[k + 1] as usize)
                .map(|c| (cyxz[3 * c], cyxz[3 * c + 1], cyxz[3 * c + 2], cbits[c], co[c]))
                .collect();
            let top: Vec<(i32, i32)> =
                (ts[k] as usize..ts[k + 1] as usize).map(|c| (txz[2 * c], txz[2 * c + 1])).collect();
            let bottom: Vec<(i32, i32)> =
                (bs[k] as usize..bs[k + 1] as usize).map(|c| (bxz[2 * c], bxz[2 * c + 1])).collect();
            let sides = (ss[k] as usize..ss[k + 1] as usize)
                .map(|c| (sv[6 * c], sv[6 * c + 1], sv[6 * c + 2], sv[6 * c + 3], sv[6 * c + 4], sv[6 * c + 5]))
                .collect();
            let halo = (hs[k] as usize..hs[k + 1] as usize)
                .map(|c| (hv[3 * c], hv[3 * c + 1], hv[3 * c + 2]))
                .collect();
            let mut top_mask = vec![false; (w * d) as usize];
            for &(a, b) in &top {
                top_mask[(a * d + b) as usize] = true;
            }
            let mut bot_mask = vec![false; (w * d) as usize];
            for &(a, b) in &bottom {
                bot_mask[(a * d + b) as usize] = true;
            }
            if bx[k] >= 0 {
                boxes.entry((bx[k], h, w, d)).or_insert(k);
            }
            opts.push(Opt {
                part: part[k] as usize,
                token: token[k] as usize,
                h,
                w,
                d,
                first: (first[3 * k], first[3 * k + 1], first[3 * k + 2]),
                logprior: lp[k],
                boxk: bx[k],
                cells,
                top,
                bottom,
                sides,
                halo,
                pix: [0, 1, 2].map(|a| {
                    (pstart[a][k] as usize..pstart[a][k + 1] as usize)
                        .map(|c| (pvals[a][2 * c], pvals[a][2 * c + 1]))
                        .collect()
                }),
                top_mask,
                bot_mask,
            });
        }
        let opt_variants = v(p.opt_variants, n_opt);
        let mut variants: std::collections::HashMap<i32, Vec<usize>> = std::collections::HashMap::new();
        for (k, &g) in opt_variants.iter().enumerate() {
            if g >= 0 { variants.entry(g).or_default().push(k); }
        }
        let n_col = p.n_col as usize;
        let n_tok = p.n_tok as usize;
        let mut d = Data {
            h: p.h,
            x: p.x,
            z: p.z,
            tbits,
            fill,
            want_cells,
            want,
            visible: v(p.visible, n).iter().map(|&b| b != 0).collect(),
            tcolour: v(p.tcolour, n),
            tlab: v(p.tlab, n * 3),
            n_col,
            pal_lab: v(p.pal_lab, n_col * 3),
            n_tok,
            nll: v(p.nll, n_tok * n_tok * 3),
            logp: v(p.logp, n_tok * n_tok * 3),
            snap: v(p.snap, p.n_part as usize * n_col),
            opts,
            boxes,
            variants,
            opt_variants,
            w_cov: p.w_cov,
            w_col: p.w_col,
            w_loc: p.w_loc,
            w_glob: p.w_glob,
            w_sup: p.w_sup,
            w_side: p.w_side,
            n_ref: p.n_ref,
            w_dens: p.w_dens,
            dens_sd: p.dens_sd,
            p_ref: p.p_ref,
            w_cnt: p.w_cnt,
            cnt_sd: p.cnt_sd,
            w_sil: p.w_sil,
            style: p.style,
            w_glob2: p.w_glob2,
            logp2: v(p.logp2, n_tok * n_tok * 3),
            middle_w: v(p.middle_w, n_tok),
            w_glob3: p.w_glob3,
            logp3: if p.logp3.is_null() { Vec::new() }
                     else { v(p.logp3, n_tok * n_tok * 3 * PATHS4_BUCKETS) },
            mid_bucket: if p.mid_bucket.is_null() { Vec::new() }
                          else { v(p.mid_bucket, n_tok * n_tok) },
            max_deg_paths4: p.max_deg_paths4,
            paths4_sample_n: p.paths4_sample_n.max(1) as u32,
            w_entropy_glob3: p.w_entropy_glob3,
            w_revkl_glob3: p.w_revkl_glob3,
            pref3: if p.pref3.is_null() { Vec::new() }
                    else { v(p.pref3, n_tok * n_tok * 3 * PATHS4_BUCKETS) },
            tsil: [
                v(p.tsx, (p.h * SUB * p.z * SUB) as usize).iter().map(|&b| b != 0).collect(),
                v(p.tsy, (p.x * SUB * p.z * SUB) as usize).iter().map(|&b| b != 0).collect(),
                v(p.tsz, (p.h * SUB * p.x * SUB) as usize).iter().map(|&b| b != 0).collect(),
            ],
            scols: [p.z * SUB, p.z * SUB, p.x * SUB],
            n_conn: p.n_conn as usize,
            fits: v(p.fits, (p.n_conn * p.n_conn) as usize),
            net: if p.use_net != 0 { Some(Net::from(p, n_opt)) } else { None },
            top_k: p.top_k.max(1) as usize,
            hinge_a: Vec::new(),
            hinge_b: Vec::new(),
            hinge_pose: Vec::new(),
            hinge_by_a: std::collections::HashMap::new(),
            region_label: if p.region_label.is_null() { Vec::new() }
                          else { v(p.region_label, n) },
            w_size: if p.opt_class.is_null() { 0.0 } else { p.w_size },
            opt_class: if p.opt_class.is_null() { Vec::new() } else { v(p.opt_class, n_opt).iter().map(|&c| c as usize).collect() },
            size_lpref: if p.size_pref.is_null() { Vec::new() }
                        else { v(p.size_pref, p.n_size_class as usize).iter().map(|&q| q.max(1e-6).ln()).collect() },
            snot_rows: Vec::new(),
            snot_by_opt: Vec::new(),
            w_sideways: if p.opt_sideways.is_null() { 0.0 } else { p.w_sideways },
            p_sideways: p.p_sideways.clamp(1e-4, 1.0 - 1e-4),
            opt_sideways: if p.opt_sideways.is_null() { Vec::new() } else { v(p.opt_sideways, n_opt).iter().map(|&b| b != 0).collect() },
            w_out: if p.w_out.is_null() || p.w_in.is_null() { Vec::new() } else { v(p.w_out, n * 64) },
            w_in: if p.w_out.is_null() || p.w_in.is_null() { Vec::new() } else { v(p.w_in, n * 64) },
            fid0: 0.0,
        };
        d.fid0 = if d.w_in.is_empty() { d.fill.iter().sum() }
                 else { (0..n).map(|c| (0..64).filter(|&b| d.tbits[c] >> b & 1 == 1).map(|b| d.w_in[c * 64 + b] as f64).sum::<f64>()).sum::<f64>() / NSAMP };
        let n_snot = p.n_snot_rows.max(0) as usize;
        if n_snot > 0 && !p.snot_rows.is_null() {
            let raw = v(p.snot_rows, n_snot * 7);
            d.snot_by_opt = vec![Vec::new(); n_opt];
            for r in 0..n_snot {
                let q = &raw[7 * r..7 * r + 7];
                d.snot_by_opt[q[0] as usize].push(d.snot_rows.len());
                d.snot_rows.push((q[1] as usize, q[2] as usize, q[3], q[4], q[5], q[6]));
            }
        }
        let n_hinge = p.n_hinge.max(0) as usize;
        if n_hinge > 0 {
            let raw = v(p.hinge_data, n_hinge * 5);
            for k in 0..n_hinge {
                let a = raw[5 * k] as usize;
                let b = raw[5 * k + 1] as usize;
                let dy = raw[5 * k + 2];
                let dx = raw[5 * k + 3];
                let dz = raw[5 * k + 4];
                if a >= d.opts.len() || b >= d.opts.len() {
                    continue;
                }
                d.hinge_by_a.entry(a).or_default().push(d.hinge_a.len());
                d.hinge_a.push(a);
                d.hinge_b.push(b);
                d.hinge_pose.push([dy, dx, dz]);
            }
        }
        d
    }

    #[inline]
    fn idx(&self, y: i32, x: i32, z: i32) -> usize {
        ((y * self.x + x) * self.z + z) as usize
    }

    #[inline]
    fn key(&self, a: usize, b: usize, d: u8) -> usize {
        (a * self.n_tok + b) * 3 + d as usize
    }

    /// 4-walk pattern key: (a_tok, d_tok, direction, mid_bucket) into a dense array.
    #[inline]
    fn key4(&self, a: usize, d_tok: usize, dir: u8, bucket: usize) -> usize {
        ((a * self.n_tok + d_tok) * 3 + dir as usize) * PATHS4_BUCKETS + bucket
    }

    /// Bucket for a middle (b_tok, c_tok) pair.  Returns -1 (interpreted as
    /// "disabled") when mid_bucket wasn't provided.
    #[inline]
    fn bucket(&self, b: usize, c: usize) -> i32 {
        if self.mid_bucket.is_empty() {
            -1
        } else {
            self.mid_bucket[b * self.n_tok + c]
        }
    }

    #[inline]
    fn paths4_enabled(&self) -> bool {
        !self.logp3.is_empty() && !self.mid_bucket.is_empty()
    }

    /// Index into view `a` of an option's outline pixel (u, v) placed at (oy, ox, oz).
    #[inline]
    fn pix_idx(&self, a: usize, u: i32, v: i32, oy: i32, ox: i32, oz: i32) -> usize {
        let (r, c) = match a {
            0 => (u + SUB * oy, v + SUB * oz),
            1 => (u + SUB * ox, v + SUB * oz),
            _ => (u + SUB * oy, v + SUB * ox),
        };
        (r * self.scols[a] + c) as usize
    }

    /// Colour error of the cells that show, if this option took this colour there.
    fn colour_cost(&self, o: &Opt, oy: i32, ox: i32, oz: i32, colour: i32) -> f64 {
        let lab = &self.pal_lab[3 * colour as usize..3 * colour as usize + 3];
        let mut s = 0.0;
        for &(y, x, z, _, occ) in &o.cells {
            let i = self.idx(oy + y, ox + x, oz + z);
            if self.visible[i] && self.tcolour[i] >= 0 {
                let t = &self.tlab[3 * i..3 * i + 3];
                let (a, b, c) = ((t[0] - lab[0]) as f64, (t[1] - lab[1]) as f64, (t[2] - lab[2]) as f64);
                s += (a * a + b * b + c * c).sqrt() * occ;
            }
        }
        s
    }

    /// Change in wrong samples (/64) from filling these empty cells so.
    fn fid_delta(&self, o: &Opt, oy: i32, ox: i32, oz: i32) -> f64 {
        if !self.w_in.is_empty() {                       // surface fit: weighted by the distance to the target's surface
            let mut s = 0f64;
            for &(y, x, z, bits, _) in &o.cells {
                let c = self.idx(oy + y, ox + x, oz + z);
                let t = self.tbits[c];
                let mut b = bits;
                while b != 0 {
                    let k = b.trailing_zeros() as usize;
                    s += if t >> k & 1 == 1 { -(self.w_in[c * 64 + k] as f64) } else { self.w_out[c * 64 + k] as f64 };
                    b &= b - 1;
                }
            }
            return s / NSAMP;
        }
        let mut s = 0i64;
        for &(y, x, z, bits, _) in &o.cells {
            let t = self.tbits[self.idx(oy + y, ox + x, oz + z)];
            s += (bits ^ t).count_ones() as i64 - t.count_ones() as i64;
        }
        s as f64 / NSAMP
    }

    /// The cells a side connector faces: just outside its face, one or (on
    /// a cell boundary) two along it (sampler.side_cells).
    fn side_cells(o: &Opt, oy: i32, ox: i32, oz: i32, sd: &(i32, i32, i32, i32, i32, i32)) -> Vec<(i32, i32, i32)> {
        let &(hx, hy, hz, dx, dz, _) = sd;
        let y = oy + (hy / 2).min(o.h - 1);
        let mut out = Vec::with_capacity(2);
        if dx != 0 {
            let x = ox + if dx > 0 { o.w } else { -1 };
            for zz in if hz % 2 == 0 { [(hz - 1).div_euclid(2), hz / 2] } else { [hz / 2, hz / 2] } {
                if zz >= 0 && zz < o.d && !out.contains(&(y, x, oz + zz)) {
                    out.push((y, x, oz + zz));
                }
            }
        } else {
            let z = oz + if dz > 0 { o.d } else { -1 };
            for xx in if hx % 2 == 0 { [(hx - 1).div_euclid(2), hx / 2] } else { [hx / 2, hx / 2] } {
                if xx >= 0 && xx < o.w && !out.contains(&(y, ox + xx, z)) {
                    out.push((y, ox + xx, z));
                }
            }
        }
        out
    }

    fn inside(&self, y: i32, x: i32, z: i32) -> bool {
        y >= 0 && x >= 0 && z >= 0 && y < self.h && x < self.x && z < self.z
    }

    /// Its side connectors facing no cell of the object.
    fn bare_sides(&self, o: &Opt, oy: i32, ox: i32, oz: i32) -> i64 {
        let mut n = 0;
        for sd in &o.sides {
            let cells = Data::side_cells(o, oy, ox, oz, sd);
            if !cells.iter().any(|&(y, x, z)| self.inside(y, x, z) && self.fill[self.idx(y, x, z)] >= 0.5) {
                n += 1;
            }
        }
        n
    }

    /// Whether a side connector of a meets one of b (sampler.sides_join).
    fn sides_join(&self, a: &Opt, ay: i32, ax: i32, az: i32, b: &Opt, by: i32, bx: i32, bz: i32) -> bool {
        for &(hx, hy, hz, dx, dz, k) in &a.sides {
            if k < 0 {
                continue;
            }
            let (px, py, pz) = (2 * ax + hx, 2 * ay + hy, 2 * az + hz);
            for &(gx, gy, gz, ex, ez, kb) in &b.sides {
                if kb >= 0
                    && self.fits[k as usize * self.n_conn + kb as usize] != 0
                    && ex == -dx
                    && ez == -dz
                    && 2 * bx + gx == px
                    && 2 * bz + gz == pz
                    && (2 * by + gy - py).abs() <= 1
                {
                    return true;
                }
            }
        }
        false
    }

    /// The share of its top with more to build above but no stud there.
    fn capped(&self, o: &Opt, oy: i32, ox: i32, oz: i32) -> f64 {
        if oy + o.h >= self.h {
            return 0.0;
        }
        let mut n = 0;
        for dx in 0..o.w {
            for dz in 0..o.d {
                if self.fill[self.idx(oy + o.h, ox + dx, oz + dz)] >= 0.5 && !o.top_mask[(dx * o.d + dz) as usize] {
                    n += 1;
                }
            }
        }
        n as f64 / (o.w * o.d) as f64
    }

    /// The target's commonest colour over its cells, as the part is made in it.
    fn colour_for(&self, o: &Opt, oy: i32, ox: i32, oz: i32, counts: &mut Vec<u32>) -> i32 {
        counts.iter_mut().for_each(|c| *c = 0);
        let mut any = false;
        for &(y, x, z, _, _) in &o.cells {
            let c = self.tcolour[self.idx(oy + y, ox + x, oz + z)];
            if c >= 0 {
                counts[c as usize] += 1;
                any = true;
            }
        }
        let mut best = 0usize;
        if any {
            for c in 0..self.n_col {
                if counts[c] > counts[best] {
                    best = c;
                }
            }
        }
        self.snap[o.part * self.n_col + best]
    }
}

// ---------------------------------------------------------------------------
// the proposal network (everything2lego/proposals.py: the same window,
// features and layers)
// ---------------------------------------------------------------------------

const WY: (i32, i32) = (-1, 4);
const WX: (i32, i32) = (-2, 4);
const WZ: (i32, i32) = (-2, 4);

struct Net {
    n_h: usize,
    n_c: usize,
    kinds: usize,
    objects: usize,
    object: i32,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    mu: Vec<f32>,
    sd: Vec<f32>,
    opt_out: Vec<i32>,
    opt_lpart: Vec<f64>,
    opt_kind: Vec<i32>,
    region_object: Vec<i32>,
    n_static: usize,
    static_: Vec<f32>,
}

impl Net {
    unsafe fn from(p: &Problem, n_opt: usize) -> Net {
        let (i, h, c) = (p.net_in as usize, p.net_h as usize, p.net_c as usize);
        Net {
            n_h: h,
            n_c: c,
            kinds: p.net_kinds as usize,
            objects: p.net_objects as usize,
            object: p.net_object,
            w1: v(p.w1, i * h),
            b1: v(p.b1, h),
            w2: v(p.w2, h * c),
            b2: v(p.b2, c),
            mu: v(p.mu, i),
            sd: v(p.sd, i),
            opt_out: v(p.opt_out, n_opt),
            opt_lpart: v(p.opt_lpart, n_opt),
            opt_kind: v(p.opt_kind, n_opt),
            region_object: v(p.region_object, (p.h * p.x * p.z) as usize),
            n_static: p.net_static as usize,
            static_: v(p.static_, (p.h * p.x * p.z) as usize * p.net_static as usize),
        }
    }

    /// log P(output | x), x the raw features.
    fn logprobs(&self, x: &[f32]) -> Vec<f64> {
        let mut h = self.b1.clone();
        for (i, &xi) in x.iter().enumerate() {
            let xn = (xi - self.mu[i]) / self.sd[i];
            if xn != 0.0 {
                let row = &self.w1[i * self.n_h..(i + 1) * self.n_h];
                for (hj, w) in h.iter_mut().zip(row) {
                    *hj += xn * w;
                }
            }
        }
        let mut z = self.b2.clone();
        for (j, &hj) in h.iter().enumerate() {
            if hj > 0.0 {
                let row = &self.w2[j * self.n_c..(j + 1) * self.n_c];
                for (zk, w) in z.iter_mut().zip(row) {
                    *zk += hj * w;
                }
            }
        }
        let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = m + z.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln();
        z.iter().map(|&v| v as f64 - lse).collect()
    }
}

/// The cell's context: the target's cells in the window, the cells built
/// before it in fill order, the kinds built there, the object's kind.
fn net_features(d: &Data, net: &Net, kindgrid: &[i32], y: i32, x: i32, z: i32) -> Vec<f32> {
    let nwin = ((WY.1 - WY.0) * (WX.1 - WX.0) * (WZ.1 - WZ.0)) as usize;
    let mut f = vec![0.0f32; 2 * nwin + net.kinds + 1 + net.objects + net.n_static];
    let mut k = 0;
    for dy in WY.0..WY.1 {
        for dx in WX.0..WX.1 {
            for dz in WZ.0..WZ.1 {
                let (yy, xx, zz) = (y + dy, x + dx, z + dz);
                if yy >= 0 && xx >= 0 && zz >= 0 && yy < d.h && xx < d.x && zz < d.z {
                    let c = d.idx(yy, xx, zz);
                    f[k] = d.fill[c] as f32;
                    let before = dy < 0 || (dy == 0 && (dx < 0 || (dx == 0 && dz < 0)));
                    let kd = kindgrid[c];
                    if before && kd >= 0 {
                        f[nwin + k] = 1.0;
                        f[2 * nwin + kd as usize] += 1.0 / nwin as f32;
                    }
                }
                k += 1;
            }
        }
    }
    let obj = {
        let r = net.region_object[d.idx(y, x, z) as usize];
        if r >= 0 { r } else { net.object }
    };
    if obj >= 0 {
        f[2 * nwin + net.kinds + 1 + obj as usize] = 1.0;
    }
    let base = 2 * nwin + net.kinds + 1 + net.objects;
    let c = d.idx(y, x, z) * net.n_static;
    f[base..base + net.n_static].copy_from_slice(&net.static_[c..c + net.n_static]);
    f
}

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

/// KL(counts || reference) under count changes: sum c log c, sum c log p, n.
/// Optionally also tracks s_plogc = sum_k p_ref(k) * log(c[k] + EPS) for a
/// reverse-direction coverage term; opt-in by passing a non-empty pref vector.
struct Kl {
    counts: Vec<f64>,
    pref: Vec<f64>,         // same length as counts; empty = don't track s_plogc
    tot_pref: f64,          // sum of pref (for normalisation of the coverage term)
    s_clogc: f64,
    s_clogp: f64,
    s_plogc: f64,
    n: f64,
}

const KL_PLOGC_EPS: f64 = 1e-6;

impl Kl {
    fn new(len: usize) -> Kl {
        Kl { counts: vec![0.0; len], pref: Vec::new(), tot_pref: 0.0,
             s_clogc: 0.0, s_clogp: 0.0, s_plogc: 0.0, n: 0.0 }
    }
    fn new_with_pref(len: usize, pref: Vec<f64>) -> Kl {
        let tot = pref.iter().sum::<f64>();
        let s0 = tot * KL_PLOGC_EPS.ln();
        Kl { counts: vec![0.0; len], pref, tot_pref: tot,
             s_clogc: 0.0, s_clogp: 0.0, s_plogc: s0, n: 0.0 }
    }
    fn add(&mut self, key: usize, dc: f64, logp: f64) {
        let c0 = self.counts[key];
        let c1 = c0 + dc;
        if c0 > 0.0 {
            self.s_clogc -= c0 * c0.ln();
        }
        if c1 > 0.0 {
            self.s_clogc += c1 * c1.ln();
        }
        self.s_clogp += dc * logp;
        if !self.pref.is_empty() {
            let p = self.pref[key];
            if p != 0.0 {
                self.s_plogc += p * ((c1 + KL_PLOGC_EPS).ln() - (c0 + KL_PLOGC_EPS).ln());
            }
        }
        self.n += dc;
        self.counts[key] = c1;
    }
    fn value(&self) -> f64 {
        if self.n <= 1e-9 {
            return 0.0;
        }
        (self.s_clogc - self.s_clogp) / self.n - self.n.ln()
    }
    /// Shannon entropy of the current count distribution: H = log(n) - s_clogc / n.
    /// Reported in nats.  Used as an anti-mode-collapse regularizer.
    fn entropy(&self) -> f64 {
        if self.n <= 1e-9 {
            return 0.0;
        }
        self.n.ln() - self.s_clogc / self.n
    }
}

#[derive(Clone, Copy)]
struct Item {
    opt: usize,
    colour: i32,
    y: i32,
    x: i32,
    z: i32,
}

struct State<'a> {
    d: &'a Data,
    owner: Vec<i32>,
    cur: Vec<u64>,
    items: Vec<Option<Item>>,
    free_ids: Vec<usize>,
    live: Vec<usize>,
    live_pos: Vec<usize>,
    unsup: Vec<bool>,
    n_unsup: i64,
    kl1: Kl, // touching pairs
    kl2: Kl, // paths of two through a shared neighbour
    kl3: Kl, // 4-walks a-b-c-d, keyed by (a_tok, d_tok, direction, mid_bucket)
    adj: Vec<Vec<(usize, u8)>>,
    path_queue: Vec<(usize, f64)>,
    path4_queue: Vec<(usize, f64)>,
    csil: [Vec<i32>; 3], // parts covering each outline pixel
    kindgrid: Vec<i32>,  // the part kind built in each cell (-1 empty), the network's context
    sil: i64,            // outline pixels wrong
    loc: f64,
    col: f64,
    fid: f64,
    bare: i64,
    size_n: Vec<f64>, // live parts per size class
    n_side: f64,      // live sideways parts
    mark: Vec<u32>,
    gen: u32,
}

const UP: u8 = 0;
const DOWN: u8 = 1;
const LEVEL: u8 = 2;

fn flip(d: u8) -> u8 {
    match d {
        UP => DOWN,
        DOWN => UP,
        _ => LEVEL,
    }
}

impl<'a> State<'a> {
    fn new(d: &'a Data) -> State<'a> {
        let n = d.tbits.len();
        State {
            d,
            owner: vec![-1; n],
            cur: vec![0; n],
            items: Vec::new(),
            free_ids: Vec::new(),
            live: Vec::new(),
            live_pos: Vec::new(),
            unsup: Vec::new(),
            n_unsup: 0,
            kl1: Kl::new(d.n_tok * d.n_tok * 3),
            kl2: Kl::new(d.n_tok * d.n_tok * 3),
            kl3: if d.paths4_enabled() {
                if !d.pref3.is_empty() && d.w_revkl_glob3 != 0.0 {
                    Kl::new_with_pref(d.n_tok * d.n_tok * 3 * PATHS4_BUCKETS, d.pref3.clone())
                } else {
                    Kl::new(d.n_tok * d.n_tok * 3 * PATHS4_BUCKETS)
                }
            } else { Kl::new(0) },
            adj: Vec::new(),
            path_queue: Vec::new(),
            path4_queue: Vec::new(),
            csil: [0, 1, 2].map(|a| vec![0; d.tsil[a].len()]),
            kindgrid: vec![-1; n],
            sil: d.tsil.iter().map(|t| t.iter().filter(|&&b| b).count() as i64).sum(),
            loc: 0.0,
            col: 0.0,
            fid: d.fid0,
            bare: 0,
            size_n: vec![0.0; d.size_lpref.len()],
            n_side: 0.0,
            mark: Vec::new(),
            gen: 0,
        }
    }

    fn item(&self, i: usize) -> &Item {
        self.items[i].as_ref().unwrap()
    }

    fn opt(&self, i: usize) -> &'a Opt {
        &self.d.opts[self.item(i).opt]
    }

    fn next_gen(&mut self) -> u32 {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.mark.iter_mut().for_each(|m| *m = 0);
            self.gen = 1;
        }
        if self.mark.len() < self.items.len() {
            self.mark.resize(self.items.len(), 0);
        }
        self.gen
    }

    /// Parts touching i — a cell of one next to a cell of the other
    /// (diagonals too) — and where, from the two boxes.  Symmetric, so a
    /// pair added from one side is removed from the other.
    fn neighbours(&mut self, i: usize) -> Vec<(usize, u8)> {
        let it = *self.item(i);
        let o = &self.d.opts[it.opt];
        let d = self.d;
        let g = self.next_gen();
        self.mark[i] = g;
        let mut out = Vec::new();
        for &(hy, hx, hz) in &o.halo {
            let (y, x, z) = (it.y + hy, it.x + hx, it.z + hz);
            if y < 0 || x < 0 || z < 0 || y >= d.h || x >= d.x || z >= d.z {
                continue;
            }
            let j = self.owner[d.idx(y, x, z)];
            if j < 0 || self.mark[j as usize] == g {
                continue;
            }
            let j = j as usize;
            self.mark[j] = g;
            let b = self.item(j);
            let bh = d.opts[b.opt].h;
            let dir = if b.y >= it.y + o.h {
                UP
            } else if b.y + bh <= it.y {
                DOWN
            } else {
                LEVEL
            };
            out.push((j, dir));
        }
        out
    }

    fn supported(&self, i: usize) -> bool {
        let it = self.item(i);
        if it.y == 0 {
            return true;
        }
        let d = self.d;
        let o = &d.opts[it.opt];
        for &(dx, dz) in &o.bottom {
            let j = self.owner[d.idx(it.y - 1, it.x + dx, it.z + dz)];
            if j >= 0 {
                let b = self.item(j as usize);
                let bo = &d.opts[b.opt];
                if b.y + bo.h == it.y && bo.has_top(it.x + dx - b.x, it.z + dz - b.z) {
                    return true;
                }
            }
        }
        if it.y + o.h < d.h {
            for &(dx, dz) in &o.top {
                let j = self.owner[d.idx(it.y + o.h, it.x + dx, it.z + dz)];
                if j >= 0 {
                    let b = self.item(j as usize);
                    let bo = &d.opts[b.opt];
                    if b.y == it.y + o.h && bo.has_bottom(it.x + dx - b.x, it.z + dz - b.z) {
                        return true;
                    }
                }
            }
        }
        self.side_held(&d.opts[it.opt], it.y, it.x, it.z, i as i32)
    }

    /// Held at a side: a pin, stud, bar or axle of it meets one of a part beside it.
    fn side_held(&self, o: &Opt, oy: i32, ox: i32, oz: i32, me: i32) -> bool {
        let d = self.d;
        for sd in &o.sides {
            for (y, x, z) in Data::side_cells(o, oy, ox, oz, sd) {
                if d.inside(y, x, z) {
                    let j = self.owner[d.idx(y, x, z)];
                    if j >= 0 && j != me {
                        let b = self.item(j as usize);
                        if d.sides_join(o, oy, ox, oz, &d.opts[b.opt], b.y, b.x, b.z) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn pairs(&mut self, i: usize, nb: &[(usize, u8)], sign: f64) {
        let ti = self.opt(i).token;
        for &(j, dir) in nb {
            let tj = self.opt(j).token;
            for (a, b, dd) in [(ti, tj, dir), (tj, ti, flip(dir))] {
                let k = self.d.key(a, b, dd);
                self.kl1.add(k, sign, self.d.logp[k]);
                self.loc += sign * self.d.nll[k];
            }
        }
    }

    /// Where b is from a, from the two boxes (features.vertical).
    fn where_(&self, a: usize, b: usize) -> u8 {
        let (ia, ib) = (self.item(a), self.item(b));
        let (ha, hb) = (self.d.opts[ia.opt].h, self.d.opts[ib.opt].h);
        if ib.y >= ia.y + ha {
            UP
        } else if ib.y + hb <= ia.y {
            DOWN
        } else {
            LEVEL
        }
    }

    /// Queue adding or removing the paths a - b - c that i is in: as the
    /// middle (every two of its neighbours) and as an end (through each
    /// neighbour b, to b's other neighbours), each weighing b's middle
    /// weight (style.middle_weight); adjacency without i on the far side.
    /// Applied merged by pattern.
    fn paths(&mut self, i: usize, nb: &[(usize, u8)], sign: f64) {
        let d = self.d;
        let w = sign * d.middle_w[self.opt(i).token];
        for &(a, _) in nb {
            for &(c, _) in nb {
                if a != c {
                    let key = d.key(self.opt(a).token, self.opt(c).token, self.where_(a, c));
                    self.path_queue.push((key, w));
                }
            }
        }
        let ti = self.opt(i).token;
        for &(b, _) in nb {
            let wb = sign * d.middle_w[self.opt(b).token];
            for k in 0..self.adj[b].len() {
                let c = self.adj[b][k].0;
                if c != i {
                    let tc = self.opt(c).token;
                    let k1 = d.key(ti, tc, self.where_(i, c));
                    let k2 = d.key(tc, ti, self.where_(c, i));
                    self.path_queue.push((k1, wb));
                    self.path_queue.push((k2, wb));
                }
            }
        }
        self.flush_paths();
    }

    fn flush_paths(&mut self) {
        let mut q = std::mem::take(&mut self.path_queue);
        q.sort_unstable_by_key(|&(k, _)| k);
        let mut i = 0;
        while i < q.len() {
            let key = q[i].0;
            let mut dc = 0.0;
            while i < q.len() && q[i].0 == key {
                dc += q[i].1;
                i += 1;
            }
            if dc != 0.0 {
                self.kl2.add(key, dc, self.d.logp2[key]);
            }
        }
        q.clear();
        self.path_queue = q;
    }

    /// 4-walks a-b-c-d that i is in.  Four positional roles enumerated; closed
    /// walks (a=d=i) counted only in position-1.  Must be called AFTER adj is
    /// updated with i (so adj[c] / adj[b] see i for closed walks).  Mirror of
    /// Python State._paths4 in src/everything2lego/sampler.py.
    fn paths4(&mut self, i: usize, nb: &[(usize, u8)], sign: f64) {
        if !self.d.paths4_enabled() {
            return;
        }
        let d = self.d;
        // Degree cap: parts whose live degree exceeds this skip their paths4 share.
        // Applied to BOTH the central part i (via nb.len()) and each middle neighbour,
        // so baseplates don't blow up the O(deg^3) enumeration.
        let cap = d.max_deg_paths4;
        if cap > 0 && nb.len() as i32 > cap {
            return;
        }
        let ti = self.opt(i).token;
        let wi = d.middle_w[ti];
        // Subsample weight scalar: when sample_n > 1, each kept walk is reweighted
        // by sample_n so the expected count matches the full enumeration.
        let sample_n = d.paths4_sample_n;
        let sample_scale = sample_n as f64;
        // Collect i's token's view of neighbours
        // Deterministic walk-hash: identical for same directed (a, b, c, d) regardless
        // of which position of i we're enumerating from.  Used for subsampling so add()
        // and remove() keep the same walks.
        let keep = |a: usize, b: usize, c: usize, dp: usize| -> bool {
            if sample_n <= 1 { return true; }
            let mut h = a as u64;
            h = h.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(b as u64);
            h = h.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(c as u64);
            h = h.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(dp as u64);
            (h % sample_n as u64) == 0
        };
        // position 2: i = b (middle-left).  middle edge (i, c), c in nb_i.
        for &(c, _) in nb {
            if cap > 0 && self.adj[c].len() as i32 > cap { continue; }
            let tc = self.opt(c).token;
            let bucket = d.bucket(ti, tc);
            if bucket < 0 { continue; }
            let w = sign * wi * d.middle_w[tc] * sample_scale;
            for k in 0..self.adj[c].len() {
                let d_part = self.adj[c][k].0;
                if d_part == i { continue; }
                let td = self.opt(d_part).token;
                for &(a, _) in nb {
                    if a == c { continue; }
                    if !keep(a, i, c, d_part) { continue; }
                    let ta = self.opt(a).token;
                    let key = d.key4(ta, td, self.where_(a, d_part), bucket as usize);
                    self.path4_queue.push((key, w));
                }
            }
        }
        // position 3: i = c (middle-right).  middle edge (b, i), b in nb_i.
        for &(b, _) in nb {
            if cap > 0 && self.adj[b].len() as i32 > cap { continue; }
            let tb = self.opt(b).token;
            let bucket = d.bucket(tb, ti);
            if bucket < 0 { continue; }
            let w = sign * d.middle_w[tb] * wi * sample_scale;
            for k in 0..self.adj[b].len() {
                let a = self.adj[b][k].0;
                if a == i { continue; }
                let ta = self.opt(a).token;
                for &(d_part, _) in nb {
                    if d_part == b { continue; }
                    if !keep(a, b, i, d_part) { continue; }
                    let td = self.opt(d_part).token;
                    let key = d.key4(ta, td, self.where_(a, d_part), bucket as usize);
                    self.path4_queue.push((key, w));
                }
            }
        }
        // position 1: i = a (walk starts at i).  Closed walks (d == i) counted here.
        for &(b, _) in nb {
            if cap > 0 && self.adj[b].len() as i32 > cap { continue; }
            let tb = self.opt(b).token;
            for k in 0..self.adj[b].len() {
                let c = self.adj[b][k].0;
                if c == i { continue; }
                if cap > 0 && self.adj[c].len() as i32 > cap { continue; }
                let tc = self.opt(c).token;
                let bucket = d.bucket(tb, tc);
                if bucket < 0 { continue; }
                let w = sign * d.middle_w[tb] * d.middle_w[tc] * sample_scale;
                for m in 0..self.adj[c].len() {
                    let d_part = self.adj[c][m].0;
                    if d_part == b { continue; }
                    if !keep(i, b, c, d_part) { continue; }
                    let td = self.opt(d_part).token;
                    let key = d.key4(ti, td, self.where_(i, d_part), bucket as usize);
                    self.path4_queue.push((key, w));
                }
            }
        }
        // position 4: i = d (walk ends at i).  Exclude a == i (closed walks already in position 1).
        for &(c, _) in nb {
            if cap > 0 && self.adj[c].len() as i32 > cap { continue; }
            let tc = self.opt(c).token;
            for k in 0..self.adj[c].len() {
                let b = self.adj[c][k].0;
                if b == i { continue; }
                if cap > 0 && self.adj[b].len() as i32 > cap { continue; }
                let tb = self.opt(b).token;
                let bucket = d.bucket(tb, tc);
                if bucket < 0 { continue; }
                let w = sign * d.middle_w[tb] * d.middle_w[tc] * sample_scale;
                for m in 0..self.adj[b].len() {
                    let a = self.adj[b][m].0;
                    if a == c || a == i { continue; }
                    if !keep(a, b, c, i) { continue; }
                    let ta = self.opt(a).token;
                    let key = d.key4(ta, ti, self.where_(a, i), bucket as usize);
                    self.path4_queue.push((key, w));
                }
            }
        }
        self.flush_paths4();
    }

    fn flush_paths4(&mut self) {
        let mut q = std::mem::take(&mut self.path4_queue);
        q.sort_unstable_by_key(|&(k, _)| k);
        let mut i = 0;
        while i < q.len() {
            let key = q[i].0;
            let mut dc = 0.0;
            while i < q.len() && q[i].0 == key {
                dc += q[i].1;
                i += 1;
            }
            if dc != 0.0 {
                self.kl3.add(key, dc, self.d.logp3[key]);
            }
        }
        q.clear();
        self.path4_queue = q;
    }

    /// Change in outline pixels wrong from adding this option here.
    fn sil_delta(&self, o: &Opt, oy: i32, ox: i32, oz: i32) -> i64 {
        let mut dlt = 0;
        for a in 0..3 {
            for &(u, v) in &o.pix[a] {
                let p = self.d.pix_idx(a, u, v, oy, ox, oz);
                if self.csil[a][p] == 0 {
                    dlt += if self.d.tsil[a][p] { -1 } else { 1 };
                }
            }
        }
        dlt
    }

    fn sil_apply(&mut self, o: &Opt, oy: i32, ox: i32, oz: i32, sign: i32) {
        for a in 0..3 {
            for &(u, v) in &o.pix[a] {
                let p = self.d.pix_idx(a, u, v, oy, ox, oz);
                let t = self.d.tsil[a][p];
                if sign > 0 {
                    if self.csil[a][p] == 0 {
                        self.sil += if t { -1 } else { 1 };
                    }
                    self.csil[a][p] += 1;
                } else {
                    self.csil[a][p] -= 1;
                    if self.csil[a][p] == 0 {
                        self.sil += if t { 1 } else { -1 };
                    }
                }
            }
        }
    }

    fn kl(&self) -> f64 {
        self.kl1.value()
    }

    fn content(&self) -> f64 {
        let d = self.d;
        d.w_cov * self.fid + d.w_sil * self.sil as f64 / (SUB * SUB) as f64 + d.w_col * self.col
    }

    fn style_loss(&self) -> f64 {
        // the pairs' typicality weighed by the pairs a real set this size
        // has, not by the build's own count (sampler.py's docstring)
        let d = self.d;
        let n = self.kl1.n;
        if n <= 0.0 {
            return 0.0;
        }
        let r = (n / d.n_ref).ln();
        let np = self.live.len() as f64;
        let rp = if np > 0.0 { (np / d.p_ref).ln() } else { 0.0 };
        let cnt = if np > 0.0 { d.w_cnt * rp * rp / (2.0 * d.cnt_sd * d.cnt_sd) } else { 0.0 };
        let kl3_v = if d.paths4_enabled() { self.kl3.value() } else { 0.0 };
        // Entropy regularizer on 4-walks: pull H(kl3) up to counter KL's mode-seeking.
        // Negative in energy means more entropy -> lower energy -> preferred.
        let ent = if d.paths4_enabled() && d.w_entropy_glob3 != 0.0 {
            -d.w_entropy_glob3 * self.kl3.entropy()
        } else { 0.0 };
        // Reverse-direction cross-entropy: -sum_k p_ref(k) * log((c[k]+eps)/n_ours).
        // = -s_plogc + tot_pref * log(n_ours).  Encourages ours to cover high-p_ref
        // keys; grows linearly with log(n_ours).  Lower value = better coverage.
        let rkl = if d.paths4_enabled() && d.w_revkl_glob3 != 0.0 && !self.kl3.pref.is_empty() && self.kl3.n > 1e-9 {
            let tot = self.kl3.tot_pref;
            d.w_revkl_glob3 * (-self.kl3.s_plogc + tot * self.kl3.n.ln())
        } else { 0.0 };
        d.style
            * d.n_ref
            * (d.w_loc * self.loc / n
                + d.w_glob * self.kl1.value()
                + d.w_glob2 * self.kl2.value()
                + d.w_glob3 * kl3_v
                + ent
                + rkl
                + d.w_dens * r * r / (2.0 * d.dens_sd * d.dens_sd)
                + cnt)
    }

    fn energy(&self) -> f64 {
        let d = self.d;
        self.content() + self.style_loss() + d.w_sup * self.n_unsup as f64 + d.w_side * self.bare as f64
            + d.w_size * self.size_kl()
            + d.w_sideways * self.sideways_kl()
    }

    /// n_parts * KL(Bernoulli(sideways share of the build) || Bernoulli(p_sideways))
    fn sideways_kl(&self) -> f64 {
        if self.d.w_sideways == 0.0 { return 0.0; }
        let n = self.live.len() as f64;
        if n <= 0.0 { return 0.0; }
        let p = self.d.p_sideways;
        let (k, m) = (self.n_side, n - self.n_side);
        (if k > 0.0 { k * (k / n / p).ln() } else { 0.0 }) + (if m > 0.0 { m * (m / n / (1.0 - p)).ln() } else { 0.0 })
    }

    /// n_parts * KL(size-class mix of the build || of real sets) = sum_c n_c ln(n_c / (n p_c))
    fn size_kl(&self) -> f64 {
        let n: f64 = self.size_n.iter().sum();
        if n <= 0.0 { return 0.0; }
        self.size_n.iter().zip(&self.d.size_lpref).filter(|(&c, _)| c > 0.0).map(|(&c, &lp)| c * ((c / n).ln() - lp)).sum()
    }

    fn set_unsup(&mut self, j: usize, v: bool) {
        if self.unsup[j] != v {
            self.unsup[j] = v;
            self.n_unsup += if v { 1 } else { -1 };
        }
    }

    fn refresh_support(&mut self, ids: &[usize]) {
        for &j in ids {
            if self.items[j].is_some() {
                let s = self.supported(j);
                self.set_unsup(j, !s);
            }
        }
    }

    fn add(&mut self, it: Item) -> usize {
        let i = match self.free_ids.pop() {
            Some(i) => i,
            None => {
                self.items.push(None);
                self.unsup.push(false);
                self.live_pos.push(0);
                self.adj.push(Vec::new());
                self.items.len() - 1
            }
        };
        self.items[i] = Some(it);
        self.live_pos[i] = self.live.len();
        self.live.push(i);
        let d = self.d;
        let o = &d.opts[it.opt];
        self.fid += d.fid_delta(o, it.y, it.x, it.z);
        if d.w_size != 0.0 { self.size_n[d.opt_class[it.opt]] += 1.0; }
        if d.w_sideways != 0.0 && d.opt_sideways[it.opt] { self.n_side += 1.0; }
        for &(y, x, z, bits, _) in &o.cells {
            let c = d.idx(it.y + y, it.x + x, it.z + z);
            self.owner[c] = i as i32;
            self.cur[c] = bits;
            if let Some(net) = &d.net {
                self.kindgrid[c] = net.opt_kind[it.opt];
            }
        }
        self.bare += d.bare_sides(o, it.y, it.x, it.z);
        self.col += d.colour_cost(o, it.y, it.x, it.z, it.colour);
        self.sil_apply(o, it.y, it.x, it.z, 1);
        let nb = self.neighbours(i);
        self.pairs(i, &nb, 1.0);
        self.paths(i, &nb, 1.0);
        for &(j, dir) in &nb {
            self.adj[j].push((i, flip(dir)));
        }
        self.adj[i] = nb.clone();
        self.paths4(i, &nb, 1.0);                                      // after adj update (closed walks need adj[c] to contain i)
        let mut ids: Vec<usize> = nb.iter().map(|&(j, _)| j).collect();
        ids.push(i);
        self.refresh_support(&ids);
        i
    }

    fn remove(&mut self, i: usize) -> Item {
        let it = *self.item(i);
        // Snapshot adj containing i for paths4 reverse-counting (closed walks).
        let nb_live: Vec<(usize, u8)> = self.adj[i].clone();
        self.paths4(i, &nb_live, -1.0);                                // before adj cleanup
        let nb = std::mem::take(&mut self.adj[i]);
        for &(j, _) in &nb {
            self.adj[j].retain(|&(k, _)| k != i);
        }
        self.paths(i, &nb, -1.0);
        self.pairs(i, &nb, -1.0);
        let d = self.d;
        let o = &d.opts[it.opt];
        self.sil_apply(o, it.y, it.x, it.z, -1);
        for &(y, x, z, _, _) in &o.cells {
            let c = d.idx(it.y + y, it.x + x, it.z + z);
            self.owner[c] = -1;
            self.cur[c] = 0;
            self.kindgrid[c] = -1;
        }
        self.bare -= d.bare_sides(o, it.y, it.x, it.z);
        self.fid -= d.fid_delta(o, it.y, it.x, it.z);
        if d.w_size != 0.0 { self.size_n[d.opt_class[it.opt]] -= 1.0; }
        if d.w_sideways != 0.0 && d.opt_sideways[it.opt] { self.n_side -= 1.0; }
        self.col -= d.colour_cost(o, it.y, it.x, it.z, it.colour);
        self.items[i] = None;
        self.set_unsup(i, false);
        let p = self.live_pos[i];
        let last = *self.live.last().unwrap();
        self.live.swap_remove(p);
        if last != i {
            self.live_pos[last] = p;
        }
        self.free_ids.push(i);
        let ids: Vec<usize> = nb.iter().map(|&(j, _)| j).collect();
        self.refresh_support(&ids);
        it
    }

    fn bad_cells(&self) -> i64 {
        self.cur
            .iter()
            .zip(&self.d.tbits)
            .filter(|(c, t)| (**c ^ **t).count_ones() >= 32)
            .count() as i64
    }
}

// ---------------------------------------------------------------------------
// proposals
// ---------------------------------------------------------------------------

struct Filler {
    colour_counts: Vec<u32>,
}

impl Filler {
    /// Whether a stud of a part already below would sit in one of its holes.
    fn held(st: &State, o: &Opt, oy: i32, ox: i32, oz: i32) -> bool {
        if oy == 0 {
            return true;
        }
        let d = st.d;
        for &(dx, dz) in &o.bottom {
            let j = st.owner[d.idx(oy - 1, ox + dx, oz + dz)];
            if j >= 0 {
                let b = st.item(j as usize);
                let bo = &d.opts[b.opt];
                if b.y + bo.h == oy && bo.has_top(ox + dx - b.x, oz + dz - b.z) {
                    return true;
                }
            }
        }
        st.side_held(o, oy, ox, oz, -1)
    }

    /// log P(part | the parts already above and below its box).
    fn stat_score(st: &mut State, k: usize, oy: i32, ox: i32, oz: i32) -> f64 {
        let d = st.d;
        let o = &d.opts[k];
        let mut s = o.logprior;
        let g = st.next_gen();
        for (yy, dir) in [(oy - 1, DOWN), (oy + o.h, UP)] {
            if yy < 0 || yy >= d.h {
                continue;
            }
            for x in ox..ox + o.w {
                for z in oz..oz + o.d {
                    let j = st.owner[d.idx(yy, x, z)];
                    if j >= 0 && st.mark[j as usize] != g {
                        st.mark[j as usize] = g;
                        let tj = d.opts[st.item(j as usize).opt].token;
                        s -= d.nll[d.key(o.token, tj, dir)];
                    }
                }
            }
        }
        s
    }

    /// Candidates whose first cell can go at (y, x, z): (pre-score, option, origin, residual).
    fn candidates(st: &State, y: i32, x: i32, z: i32, prior: Option<&[f64]>,
                  out: &mut Vec<(f64, usize, i32, i32, i32, f64)>) {
        out.clear();
        let d = st.d;
        'opt: for (k, o) in d.opts.iter().enumerate() {
            let (oy, ox, oz) = (y - o.first.0, x - o.first.1, z - o.first.2);
            if oy < 0 || ox < 0 || oz < 0 || oy + o.h > d.h || ox + o.w > d.x || oz + o.d > d.z {
                continue;
            }
            let mut wrong = 0u32;
            for &(cy, cx, cz, bits, occ) in &o.cells {
                let c = d.idx(oy + cy, ox + cx, oz + cz);
                if st.owner[c] >= 0 {
                    continue 'opt;
                }
                let t = d.tbits[c];
                if occ >= 0.5 && t == 0 {
                    continue 'opt; // solid where the object is not
                }
                wrong += (bits ^ t).count_ones();
            }
            let resid = wrong as f64 / NSAMP;
            // style and content weighed as in the energy
            let lp = prior.map_or(o.logprior, |p| p[k]);
            out.push((d.style * lp - d.w_cov * resid, k, oy, ox, oz, resid));
        }
        let k = d.top_k;
        if out.len() > k {
            out.select_nth_unstable_by(k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
            out.truncate(k);
        }
    }

    fn fill(&mut self, st: &mut State, rng: &mut Rng, mut cells: Vec<(i32, i32, i32)>) -> Vec<usize> {
        cells.sort_unstable();
        cells.dedup();
        let d = st.d;
        let mut added = Vec::new();
        let mut cands = Vec::new();
        let mut scored: Vec<(f64, f64, usize, i32, i32, i32, i32)> = Vec::new();
        for (y, x, z) in cells {
            if st.owner[d.idx(y, x, z)] >= 0 {
                continue;
            }
            // with the network: log P(part, yaw | the cell's context), once per cell
            let prior: Option<Vec<f64>> = d.net.as_ref().map(|net| {
                let lp = net.logprobs(&net_features(d, net, &st.kindgrid, y, x, z));
                (0..d.opts.len()).map(|k| lp[net.opt_out[k] as usize] + net.opt_lpart[k]).collect()
            });
            Filler::candidates(st, y, x, z, prior.as_deref(), &mut cands);
            if cands.is_empty() {
                continue;
            }
            scored.clear();
            for &(_, k, oy, ox, oz, resid) in &cands {
                let o = &d.opts[k];
                let colour = d.colour_for(o, oy, ox, oz, &mut self.colour_counts);
                let stat = d.style
                    * match &prior {
                        Some(p) => p[k],
                        None => Filler::stat_score(st, k, oy, ox, oz),
                    };
                let mut fit = -d.w_cov * resid
                    - d.w_sil * st.sil_delta(o, oy, ox, oz) as f64 / (SUB * SUB) as f64
                    - d.w_col * d.colour_cost(o, oy, ox, oz, colour);
                if !Filler::held(st, o, oy, ox, oz) {
                    fit -= d.w_sup;
                }
                fit -= d.w_sup * d.capped(o, oy, ox, oz) + d.w_side * d.bare_sides(o, oy, ox, oz) as f64;
                scored.push((stat, fit, k, oy, ox, oz, colour));
            }
            let best_stat = scored.iter().map(|s| s.0).fold(f64::NEG_INFINITY, f64::max);
            let mut scores: Vec<f64> = scored.iter().map(|s| s.0 + s.1).collect();
            let f = d.fill[d.idx(y, x, z)];
            let skip = f < 0.5; // a barely-wanted cell may stay empty
            if skip {
                scores.push(best_stat - d.w_cov * f);
            }
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ws: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let total: f64 = ws.iter().sum();
            let mut r = rng.f64() * total;
            let mut pick = ws.len() - 1;
            for (i, w) in ws.iter().enumerate() {
                if r < *w {
                    pick = i;
                    break;
                }
                r -= w;
            }
            if skip && pick == scored.len() {
                continue;
            }
            let (_, _, k, oy, ox, oz, colour) = scored[pick];
            added.push(st.add(Item { opt: k, colour, y: oy, x: ox, z: oz }));
        }
        added
    }
}

enum Undo {
    Retile { added: Vec<usize>, removed: Vec<Item> },
    Merge { k: usize, a: Item, b: Item },
    Swap { k: usize, a: Item },
    Hinge { k: usize, evicted: Option<Item> },
    MasonryShift { k: usize, a: Item },
}

fn undo(st: &mut State, u: Undo) {
    match u {
        Undo::Retile { added, removed } => {
            for i in added {
                st.remove(i);
            }
            for it in removed {
                st.add(it);
            }
        }
        Undo::Merge { k, a, b } => {
            st.remove(k);
            st.add(a);
            st.add(b);
        }
        Undo::Swap { k, a } => {
            st.remove(k);
            st.add(a);
        }
        Undo::Hinge { k, evicted } => {
            st.remove(k);
            if let Some(it) = evicted {
                st.add(it);
            }
        }
        Undo::MasonryShift { k, a } => {
            st.remove(k);
            st.add(a);
        }
    }
}

/// Remove the parts meeting a random window and re-fill its cells.
fn retile(st: &mut State, filler: &mut Filler, rng: &mut Rng) -> Option<Undo> {
    let d = st.d;
    if d.want_cells.is_empty() {
        return None;
    }
    let c = d.want_cells[rng.below(d.want_cells.len())] as i32;
    let (cy, cx, cz) = (c / (d.x * d.z), (c / d.z) % d.x, c % d.z);
    let hy = if rng.f64() < 0.5 { 1 } else { 3 };
    let hx = 1 + rng.below(3) as i32;
    let hz = 1 + rng.below(3) as i32;
    let (y0, y1) = (cy, (cy + hy).min(d.h));
    let (x0, x1) = ((cx - hx + 1).max(0), (cx + hx).min(d.x));
    let (z0, z1) = ((cz - hz + 1).max(0), (cz + hz).min(d.z));
    let mut ids = Vec::new();
    let g = st.next_gen();
    for y in y0..y1 {
        for x in x0..x1 {
            for z in z0..z1 {
                let j = st.owner[d.idx(y, x, z)];
                if j >= 0 && st.mark[j as usize] != g {
                    st.mark[j as usize] = g;
                    ids.push(j as usize);
                }
            }
        }
    }
    let mut cells = Vec::new();
    let mut removed = Vec::with_capacity(ids.len());
    for i in ids {
        let it = st.remove(i);
        for &(y, x, z, _, _) in &d.opts[it.opt].cells {
            cells.push((it.y + y, it.x + x, it.z + z));
        }
        removed.push(it);
    }
    for y in y0..y1 {
        for x in x0..x1 {
            for z in z0..z1 {
                let i = d.idx(y, x, z);
                if d.want[i] && st.owner[i] < 0 {
                    cells.push((y, x, z));
                }
            }
        }
    }
    let added = filler.fill(st, rng, cells);
    Some(Undo::Retile { added, removed })
}

/// Join a random plain part with a neighbour of the same kind, height and
/// colour beside it into the one real part covering both.
fn merge(st: &mut State, rng: &mut Rng) -> Option<Undo> {
    if st.live.is_empty() {
        return None;
    }
    let i = st.live[rng.below(st.live.len())];
    let a = *st.item(i);
    let d = st.d;
    let ao = &d.opts[a.opt];
    if ao.boxk < 0 {
        return None;
    }
    // among the neighbours a can merge with, pick the largest (bigger merge =
    // one more part gone, which the density term asks for)
    let mut best: Option<(i32, usize, i32, i32, i32, i32, usize)> = None;
    for (j, dir) in st.adj[i].clone() {
        if dir != LEVEL {
            continue;
        }
        let b = *st.item(j);
        let bo = &d.opts[b.opt];
        if bo.boxk != ao.boxk || bo.h != ao.h || b.colour != a.colour || b.y != a.y {
            continue;
        }
        let (x0, z0) = (a.x.min(b.x), a.z.min(b.z));
        let (x1, z1) = ((a.x + ao.w).max(b.x + bo.w), (a.z + ao.d).max(b.z + bo.d));
        if (x1 - x0) * (z1 - z0) != ao.w * ao.d + bo.w * bo.d {
            continue;
        }
        let k = match d.boxes.get(&(ao.boxk, ao.h, x1 - x0, z1 - z0)) {
            Some(&k) => k,
            None => continue,
        };
        if d.snap[d.opts[k].part * d.n_col + a.colour as usize] != a.colour {
            continue;
        }
        let area = (x1 - x0) * (z1 - z0);
        if best.map_or(true, |(ba, ..)| area > ba) {
            best = Some((area, j, x0, z0, x1, z1, k));
        }
    }
    let (_, j, x0, z0, _x1, _z1, k) = best?;
    let ra = st.remove(i);
    let rb = st.remove(j);
    let k = st.add(Item { opt: k, colour: a.colour, y: a.y, x: x0, z: z0 });
    Some(Undo::Merge { k, a: ra, b: rb })
}

/// Replace a plain part with another of the same box kind and footprint
/// (87087 side-stud, 11211 double-side, round, tooth; sampler.swap).
fn swap(st: &mut State, rng: &mut Rng) -> Option<Undo> {
    if st.live.is_empty() {
        return None;
    }
    let i = st.live[rng.below(st.live.len())];
    let a = *st.item(i);
    let d = st.d;
    let g = d.opt_variants[a.opt];
    if g < 0 {
        return None;
    }
    let group = &d.variants[&g];
    if group.len() < 2 {
        return None;
    }
    let mut k = a.opt;
    for _ in 0..4 {
        k = group[rng.below(group.len())];
        if k != a.opt {
            break;
        }
    }
    if k == a.opt {
        return None;
    }
    if d.snap[d.opts[k].part * d.n_col + a.colour as usize] != a.colour {
        return None;                                 // not made in this colour
    }
    let ra = st.remove(i);
    let k2 = st.add(Item { opt: k, colour: a.colour, y: a.y, x: a.x, z: a.z });
    Some(Undo::Swap { k: k2, a: ra })
}

/// Attach B at its canonical pose to an existing A, drawn from the hinge-pair
/// catalogue (everything2lego/hinges.py).  Mirrors sampler.hinge_swap.
fn hinge_swap(st: &mut State, rng: &mut Rng) -> Option<Undo> {
    let d = st.d;
    if d.hinge_a.is_empty() || st.live.is_empty() {
        return None;
    }
    // sample a few live items and pick one whose opt has a hinge pair
    let mut i = usize::MAX;
    for _ in 0..8 {
        let cand = st.live[rng.below(st.live.len())];
        let a = st.item(cand);
        if d.hinge_by_a.contains_key(&a.opt) {
            i = cand;
            break;
        }
    }
    if i == usize::MAX {
        return None;
    }
    let a = *st.item(i);
    let pairs = &d.hinge_by_a[&a.opt];
    let p = pairs[rng.below(pairs.len())];
    let b_opt = d.hinge_b[p];
    let [dy, dx, dz] = d.hinge_pose[p];
    let (by, bx, bz) = (a.y + dy, a.x + dx, a.z + dz);
    let b = &d.opts[b_opt];
    // in-bounds?
    if by < 0 || by + b.h > d.h || bx < 0 || bx + b.w > d.x || bz < 0 || bz + b.d > d.z {
        return None;
    }
    // cells either empty or all owned by one evictable item (not A)
    let mut occ: Option<usize> = None;
    for &(cy, cx, cz, _, _) in &b.cells {
        let j = st.owner[d.idx(by + cy, bx + cx, bz + cz)];
        if j < 0 {
            continue;
        }
        let j = j as usize;
        if j == i {
            return None;
        }
        if let Some(k) = occ {
            if k != j {
                return None;
            }
        } else {
            occ = Some(j);
        }
    }
    let evicted = if let Some(j) = occ {
        // only evict j if all its cells are inside B's footprint (don't orphan)
        let jitem = *st.item(j);
        let jo = &d.opts[jitem.opt];
        let b_cells: std::collections::HashSet<(i32, i32, i32)> =
            b.cells.iter().map(|&(cy, cx, cz, _, _)| (by + cy, bx + cx, bz + cz)).collect();
        let ok = jo.cells.iter().all(|&(cy, cx, cz, _, _)|
            b_cells.contains(&(jitem.y + cy, jitem.x + cx, jitem.z + cz)));
        if !ok {
            return None;
        }
        Some(st.remove(j))
    } else {
        None
    };
    // colour: inherit A's unless not allowed; snap replaces a disallowed one
    let colour = d.snap[b.part * d.n_col + a.colour as usize];
    let k = st.add(Item { opt: b_opt, colour, y: by, x: bx, z: bz });
    Some(Undo::Hinge { k, evicted })
}

/// Shift a wall-region long rectangular part by 1 stud along its long axis
/// to break vertically-aligned seams with the part directly above.
/// Mirrors sampler.masonry_shift.
fn masonry_shift(st: &mut State, rng: &mut Rng) -> Option<Undo> {
    let d = st.d;
    if d.region_label.is_empty() || st.live.is_empty() {
        return None;
    }
    let i = st.live[rng.below(st.live.len())];
    let a = *st.item(i);
    let o = &d.opts[a.opt];
    // long rectangular box
    if o.w + o.d < 3 || o.w * o.d > 8 {
        return None;
    }
    // dominant region of A's cells must be wall (label 1)
    let mut counts = [0i32; 8];
    for &(cy, cx, cz, _, _) in &o.cells {
        let r = d.region_label[d.idx(a.y + cy, a.x + cx, a.z + cz)];
        if r >= 0 && (r as usize) < counts.len() {
            counts[r as usize] += 1;
        }
    }
    let dom = counts.iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0);
    if dom != 1 {
        return None;
    }
    // part B above with same (x_stud, z_stud, W, D) at y_top?
    let y_top = a.y + o.h;
    if y_top >= d.h {
        return None;
    }
    let mut above: i32 = -1;
    'find: for x in a.x..a.x + o.w {
        for z in a.z..a.z + o.d {
            let j = st.owner[d.idx(y_top, x, z)];
            if j >= 0 {
                above = j;
                break 'find;
            }
        }
    }
    if above < 0 {
        return None;
    }
    let b = *st.item(above as usize);
    let bo = &d.opts[b.opt];
    if b.y != y_top || b.x != a.x || b.z != a.z || bo.w != o.w || bo.d != o.d {
        return None;
    }
    // shift along the long axis
    let along_x = o.w >= o.d;
    for &dir in &[1i32, -1i32] {
        let nx = a.x + if along_x { dir } else { 0 };
        let nz = a.z + if along_x { 0 } else { dir };
        // in-bounds + cells free (ignoring self) + in want
        let ok = o.cells.iter().all(|&(cy, cx, cz, _, _)| {
            let yy = a.y + cy;
            let xx = nx + cx;
            let zz = nz + cz;
            if yy < 0 || yy >= d.h || xx < 0 || xx >= d.x || zz < 0 || zz >= d.z {
                return false;
            }
            let idx = d.idx(yy, xx, zz);
            d.want[idx] && (st.owner[idx] == -1 || st.owner[idx] == i as i32)
        });
        if !ok {
            continue;
        }
        let ra = st.remove(i);
        let k = st.add(Item { opt: a.opt, colour: a.colour, y: a.y, x: nx, z: nz });
        return Some(Undo::MasonryShift { k, a: ra });
    }
    None
}

/// Sideways building in one step: a plain brick becomes its side-stud version, and a sideways plate / tile is hung
/// on each of its side studs, the parts in their way removed (at most 3 per tile).  The energy takes or leaves it.
fn snot_face(st: &mut State, rng: &mut Rng) -> Option<Undo> {
    if st.live.is_empty() {
        return None;
    }
    let d = st.d;
    let i = st.live[rng.below(st.live.len())];
    let a = *st.item(i);
    let rows = &d.snot_by_opt[a.opt];
    if rows.is_empty() {
        return None;
    }
    let anchor = d.snot_rows[rows[rng.below(rows.len())]].0;
    let ra = st.remove(i);
    let k = st.add(Item { opt: anchor, colour: a.colour, y: a.y, x: a.x, z: a.z });
    let mut added = vec![k];
    let mut removed = vec![ra];
    let mut studs: Vec<i32> = Vec::new();
    for &r in rows.iter() {
        let (an, tile, dy, dx, dz, stud) = d.snot_rows[r];
        if an != anchor || studs.contains(&stud) {
            continue;
        }
        let t = &d.opts[tile];
        let (ty, tx, tz) = (a.y + dy, a.x + dx, a.z + dz);
        let mut owners: Vec<usize> = Vec::new();
        let mut ok = true;
        for &(cy, cx, cz, _, _) in &t.cells {
            let (y, x, z) = (ty + cy, tx + cx, tz + cz);
            if !d.inside(y, x, z) {
                ok = false;
                break;
            }
            let o = st.owner[d.idx(y, x, z)];
            if o >= 0 {
                let o = o as usize;
                if added.contains(&o) {
                    ok = false;
                    break;
                }
                if !owners.contains(&o) {
                    owners.push(o);
                }
            }
        }
        if !ok || owners.len() > 3 {
            continue;
        }
        let mut colour = a.colour;
        for o in owners {
            let it = st.remove(o);
            colour = it.colour;                                  // the tile takes the colour of the face it replaces
            removed.push(it);
        }
        added.push(st.add(Item { opt: tile, colour, y: ty, x: tx, z: tz }));
        studs.push(stud);
    }
    if added.len() == 1 {
        undo(st, Undo::Retile { added, removed });
        return None;
    }
    Some(Undo::Retile { added, removed })
}

fn propose(st: &mut State, filler: &mut Filler, rng: &mut Rng) -> Option<Undo> {
    if !st.d.snot_by_opt.is_empty() && rng.f64() < 0.05 {        // only drawn when sideways building is on
        if let Some(u) = snot_face(st, rng) {
            return Some(u);
        }
    }
    let r = rng.f64();
    if r < 0.02 { masonry_shift(st, rng) }
    else if r < 0.05 { hinge_swap(st, rng) }
    else if r < 0.12 { swap(st, rng) }
    else if r < 0.57 { merge(st, rng) }
    else { retile(st, filler, rng) }
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

unsafe fn run(p: &Problem) -> i32 {
    let d = Data::from(p);
    let mut st = State::new(&d);
    let mut rng = Rng::new(p.seed);
    let mut filler = Filler { colour_counts: vec![0; d.n_col.max(1)] };
    // multi-scale warm-start: pre-populate state with coarse placements
    let n_init = p.n_init.max(0) as usize;
    if n_init > 0 {
        let init_opt = v(p.init_opt, n_init);
        let init_colour = v(p.init_colour, n_init);
        let init_y = v(p.init_y, n_init);
        let init_x = v(p.init_x, n_init);
        let init_z = v(p.init_z, n_init);
        for i in 0..n_init {
            let opt = init_opt[i] as usize;
            if opt >= d.opts.len() {
                continue;
            }
            let o = &d.opts[opt];
            let (y, x, z) = (init_y[i], init_x[i], init_z[i]);
            let ok = o.cells.iter().all(|&(cy, cx, cz, _, _)| {
                let yy = y + cy;
                let xx = x + cx;
                let zz = z + cz;
                yy >= 0 && yy < d.h && xx >= 0 && xx < d.x && zz >= 0 && zz < d.z
                    && d.want[d.idx(yy, xx, zz)]
                    && st.owner[d.idx(yy, xx, zz)] < 0
            });
            if ok {
                st.add(Item { opt, colour: init_colour[i], y, x, z });
            }
        }
    }
    let cells: Vec<(i32, i32, i32)> = d
        .want_cells
        .iter()
        .map(|&c| {
            let c = c as i32;
            (c / (d.x * d.z), (c / d.z) % d.x, c % d.z)
        })
        .filter(|&(y, x, z)| st.owner[d.idx(y, x, z)] < 0)
        .collect();
    filler.fill(&mut st, &mut rng, cells);
    let mut e = st.energy();
    let mut history = vec![e];
    let mut deltas = Vec::new();
    for _ in 0..p.warmup {
        // measure the scale, change nothing
        if let Some(u) = propose(&mut st, &mut filler, &mut rng) {
            deltas.push((st.energy() - e).abs());
            undo(&mut st, u);
        }
    }
    let mut pos: Vec<f64> = deltas.into_iter().filter(|&x| x > 0.0).collect();
    let scale = if pos.is_empty() {
        1.0
    } else {
        pos.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = pos.len();
        if n % 2 == 1 {
            pos[n / 2]
        } else {
            0.5 * (pos[n / 2 - 1] + pos[n / 2])
        }
    };
    let (t_start, t_final) = (scale * p.t_start_share, scale * p.t_final_share);
    let mut accepted = 0i64;
    let iters = p.iters.max(0);
    for it in 0..iters {
        let t = t_start * (t_final / t_start).powf(it as f64 / (iters - 1).max(1) as f64);
        let u = match propose(&mut st, &mut filler, &mut rng) {
            Some(u) => u,
            None => continue,
        };
        let e2 = st.energy();
        if e2 <= e || rng.f64() < (-(e2 - e) / t).exp() {
            e = e2;
            accepted += 1;
        } else {
            undo(&mut st, u);
        }
        if p.log_every > 0 && (it + 1) % p.log_every == 0 {
            history.push(e);
        }
    }
    // outputs
    let n = st.live.len().min(p.out_cap as usize);
    for (k, &i) in st.live.iter().take(n).enumerate() {
        let it = st.item(i);
        *p.out_opt.add(k) = it.opt as i32;
        *p.out_colour.add(k) = it.colour;
        *p.out_y.add(k) = it.y;
        *p.out_x.add(k) = it.x;
        *p.out_z.add(k) = it.z;
    }
    *p.out_n = n as i32;
    let kl = st.kl();
    let info = [
        e,
        accepted as f64 / iters.max(1) as f64,
        t_start,
        kl,
        st.loc / st.kl1.n.max(1.0),
        st.fid,
        st.bad_cells() as f64,
        st.n_unsup as f64,
        st.bare as f64,
        st.live.len() as f64,
        st.kl1.n,
        {
            // from scratch: every part re-added to an empty state
            let mut fresh = State::new(&d);
            let all: Vec<Item> = st.live.iter().map(|&i| *st.item(i)).collect();
            for it in all {
                fresh.add(it);
            }
            fresh.energy()
        },
        st.kl2.value(),
        st.sil as f64 / (SUB * SUB) as f64,
        st.content(),
        st.style_loss(),
        if d.paths4_enabled() { st.kl3.value() } else { 0.0 },
        if d.paths4_enabled() { st.kl3.entropy() } else { 0.0 },
        if d.paths4_enabled() && !st.kl3.pref.is_empty() && st.kl3.n > 1e-9 {
            -st.kl3.s_plogc + st.kl3.tot_pref * st.kl3.n.ln()
        } else { 0.0 },
        st.size_kl() / (st.live.len().max(1) as f64),                  // size-class KL to real sets
    ];
    for (k, v) in info.iter().enumerate().take(INFO_LEN) {
        *p.info.add(k) = *v;
    }
    let hn = history.len().min(p.hist_cap as usize);
    for (k, v) in history.iter().take(hn).enumerate() {
        *p.hist.add(k) = *v;
    }
    *p.hist_n = hn as i32;
    if st.live.len() > p.out_cap as usize {
        return 2; // output truncated
    }
    0
}

/// Run the sampler.  0: done; 1: it panicked; 2: more parts than `out_cap`.
#[no_mangle]
pub unsafe extern "C" fn e2l_sample(p: *const Problem) -> i32 {
    let p = &*p;
    match panic::catch_unwind(|| run(p)) {
        Ok(r) => r,
        Err(_) => 1,
    }
}

/// The layout version: Python checks it before calling.
#[no_mangle]
pub extern "C" fn e2l_abi() -> i32 {
    16
}
