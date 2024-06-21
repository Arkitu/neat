use std::f32::consts::E;
type F = f32;

fn sigmoid(x: F) -> F {
    1. / (1. + E.powf(-x))
}

struct Node {
    bias: F,
    sources: Vec<(F, usize)>
}

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
        vals[vals.len()-OUT-1..].try_into().unwrap()
    }
}

struct Neuron {
    evo_num: usize,
    bias: F
}
struct Link {
    evo_num: usize,
    strength: F,
    in_neuron: usize, // evo_num
    out_neuron: usize // evo_num
}

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
}

struct Trainer<const IN: usize, const OUT: usize, const GENERATION_SIZE: usize = 50> {
    score_fn: fn([Nn<IN, OUT>; GENERATION_SIZE]) -> [F; GENERATION_SIZE],
    current_gen: Option<[Phenotype<IN, OUT>; GENERATION_SIZE]>,
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
        let gen = self.current_gen.unwrap_or([]);
    }
}

fn main() {
    let trainer = Trainer::<2, 1>::new(|gen| gen.map(|nn| {
        dbg!(-nn.eval([0., 0.])[0]+nn.eval([1., 0.])[0]+nn.eval([0., 1.])[0]-nn.eval([1., 1.])[0])
    }));
}
