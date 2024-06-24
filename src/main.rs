use std::array;
use rand::prelude::*;

type F = f32;
const E: F = std::f32::consts::E;

fn sigmoid(x: F) -> F {
    1. / (1. + E.powf(-x))
}

#[derive(Debug)]
struct Node {
    bias: F,
    sources: Vec<(F, usize)>
}

#[derive(Debug)]
struct Nn<const IN: usize, const OUT: usize> {
    nodes: Vec<Node>
}
impl<const IN: usize, const OUT: usize> Nn<IN, OUT> {
    fn eval(&self, input: [F; IN]) -> [F; OUT] {
        let mut vals = Vec::with_capacity(self.nodes.len());
        for input in input.into_iter() {
            vals.push(input);
        }
        for n in self.nodes.iter().skip(IN) {
            let mut v = n.bias;
            for s in n.sources.iter() {
                v += s.0 * vals[s.1];
            }
            vals.push(sigmoid(v));
        }
        vals[vals.len()-OUT..].try_into().unwrap()
    }
}

#[derive(Debug, Clone, Copy)]
struct Neuron {
    evo_num: usize,
    bias: F
}
#[derive(Debug, Clone)]
struct Link {
    evo_num: usize,
    strength: F,
    in_neuron: usize, // evo_num
    out_neuron: usize, // evo_num
    disabled: bool
}

#[derive(Debug)]
enum Gene {
    Neuron(Neuron),
    Link(Link)
}
impl Gene {
    fn evo_num(&self) -> usize {
        match self {
            &Gene::Neuron(ref n) => n.evo_num,
            &Gene::Link(ref l) => l.evo_num
        }
    }
}

#[derive(Debug)]
struct Phenotype<const IN: usize, const OUT: usize> {
    neurons: Vec<Neuron>, // Keep in order
    links: Vec<Link>
}
impl<const IN: usize, const OUT: usize> Default for Phenotype<IN, OUT> {
    fn default() -> Self {
        let mut neurons = Vec::with_capacity(IN+OUT);
        for i in 0..IN+OUT {
            neurons.push(Neuron { evo_num: i, bias: 0. })
        }
        Self {
            neurons,
            links: Vec::new()
        }
    }
}
impl<const IN: usize, const OUT: usize> Phenotype<IN, OUT> {
    fn nn(&self) -> Nn<IN, OUT> {
        let mut nodes = self.neurons.iter()
            .map(|n| Node {
                bias: n.bias,
                sources: Vec::new()
            }).collect::<Vec<_>>();
        
        for l in self.links.iter() {
            nodes[self.neurons.iter().position(|n| n.evo_num == l.out_neuron).unwrap()].sources.push((l.strength, self.neurons.iter().position(|n| n.evo_num == l.in_neuron).unwrap()))
        }
        Nn {
            nodes
        }
    }
    fn mutate(&mut self, current_evo_num: &mut usize) {
        let mut rng = thread_rng();
        for n in self.neurons.iter_mut() {
            n.bias = (n.bias + rng.gen_range(-0.3..0.3)).min(1.);
        }
        for l in self.links.iter_mut() {
            l.strength = (l.strength + rng.gen_range(-0.3..0.3)).min(1.);
        }
        if self.links.len() > 0 && rng.gen_bool(0.1) {
            let l = self.links.choose_mut(&mut rng).unwrap();
            l.disabled = true;
            let l = l.clone();

            let ns = self.neurons.iter().enumerate().filter(|(_, n)| n.evo_num == l.in_neuron || n.evo_num == l.evo_num).map(|(i,n)| (i,n.evo_num)).collect::<Vec<_>>();
            assert_eq!(ns.len(), 2);

            
            self.neurons.insert(rng.gen_range((ns[0].0+1)..=ns[1].0), Neuron {
                evo_num: *current_evo_num+1,
                bias: 0.
            });

            self.links.push(Link {
                evo_num: *current_evo_num+2,
                strength: l.strength.abs().sqrt(),
                in_neuron: ns[0].1,
                out_neuron: *current_evo_num+1,
                disabled: false
            });
            self.links.push(Link {
                evo_num: *current_evo_num+3,
                strength: l.strength.abs().sqrt().copysign(l.strength),
                in_neuron: *current_evo_num+1,
                out_neuron: ns[1].1,
                disabled: false
            });

            *current_evo_num += 3;
        }
        if rng.gen_bool(0.2) {
            let n1 = self.neurons.iter().enumerate().choose(&mut rng).unwrap();
            
        }
    }
}

#[derive(Debug)]
struct Trainer<const IN: usize, const OUT: usize, const GENERATION_SIZE: usize = 50> {
    score_fn: fn([Nn<IN, OUT>; GENERATION_SIZE]) -> [F; GENERATION_SIZE],
    current_gen: Option<[(Phenotype<IN, OUT>, F); GENERATION_SIZE]>,
    current_evo_num: usize
}
impl<const IN: usize, const OUT: usize, const GENERATION_SIZE: usize> Trainer<IN, OUT, GENERATION_SIZE> {
    fn new(score_fn: fn([Nn<IN, OUT>; GENERATION_SIZE]) -> [F; GENERATION_SIZE]) -> Self {
        Self {
            score_fn,
            current_gen: None,
            current_evo_num: 0
        }
    }
    fn train(&mut self) {
        let mut gen = self.current_gen.take().unwrap_or_else(|| array::from_fn(|_| (Phenotype::default(), 0.)));
        for p in gen.iter_mut() {
            p.0.mutate(&mut self.current_evo_num);
        }
        let scores = (self.score_fn)(array::from_fn(|i| gen[i].0.nn()));
        for (i, s) in scores.into_iter().enumerate() {
            gen[i].1 = s;
        }
        gen.sort_unstable_by(|a,b| a.1.total_cmp(&b.1).reverse());
        self.current_gen = Some(gen);
    }
    fn best_score(&self) -> Option<F> {
        self.current_gen.as_ref().map(|g| g[0].1)
    }
}

fn main() {
    // let p = Phenotype {
    //     neurons: vec![Neuron]
    // }

    let mut trainer = Trainer::<2, 1>::new(|gen| gen.map(|nn| {
        dbg!(&nn);
        dbg!(-nn.eval([0., 0.])[0])+dbg!(nn.eval([1., 0.])[0])+dbg!(nn.eval([0., 1.])[0])-dbg!(nn.eval([1., 1.])[0])
    }));
    for _ in 0..10 {
        dbg!(trainer.best_score());
        trainer.train();
    }
}
