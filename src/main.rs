use std::{array, collections::HashMap, sync::{Arc, RwLock}};
use rand::{distributions::Uniform, prelude::*};
use plotters::prelude::*;
use rayon::prelude::*;

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

#[derive(Debug, Clone)]
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
    fn mutate(&mut self, cen: &mut usize) {
        let mut rng = thread_rng();
        // self.neurons.par_iter_mut().for_each_init(|| thread_rng(), |rng, n| n.bias += rng.gen_range(-0.1..0.1));
        // self.links.par_iter_mut().for_each_init(|| thread_rng(), |rng, l| l.strength += rng.gen_range(-0.1..0.1));
        for (n, r) in self.neurons.iter_mut().zip(Uniform::new(-0.1, 0.1).sample_iter(rng.clone())) {
            n.bias = (n.bias+r).clamp(-2., 2.);
        }
        for (l, r) in self.links.iter_mut().zip(Uniform::new(-0.1, 0.1).sample_iter(rng.clone())) {
            l.strength = (l.strength+r).clamp(-2., 2.);
        }
        if self.links.len() > 0 && rng.gen_bool(0.03) {
            let l = self.links.choose_mut(&mut rng).unwrap();
            l.disabled = true;
            let l = l.clone();

            let ns = self.neurons.iter().enumerate().filter(|(_, n)| n.evo_num == l.in_neuron || n.evo_num == l.out_neuron).map(|(i,n)| (i,n.evo_num)).collect::<Vec<_>>();

            self.neurons.insert(rng.gen_range((ns[0].0.max(IN-1)+1)..=ns[1].0.min(self.neurons.len()+1-OUT)), Neuron {
                evo_num: *cen+1,
                bias: 0.
            });

            self.links.push(Link {
                evo_num: *cen+2,
                strength: l.strength.abs().sqrt(),
                in_neuron: ns[0].1,
                out_neuron: *cen+1,
                disabled: false
            });
            self.links.push(Link {
                evo_num: *cen+3,
                strength: l.strength.abs().sqrt().copysign(l.strength),
                in_neuron: *cen+1,
                out_neuron: ns[1].1,
                disabled: false
            });

            *cen += 3;
        }
        if rng.gen_bool(0.05) {
            let i1 = rng.gen_range(0..(self.neurons.len()-OUT));
            let i2 = rng.gen_range((i1+1).max(IN)..self.neurons.len());
            let e1 = self.neurons[i1].evo_num;
            let e2 = self.neurons[i2].evo_num;
            if !self.links.iter().any(|l| l.in_neuron == e1 && l.out_neuron == e2) {
                self.links.push(Link {
                    evo_num: *cen+1,
                    strength: 0.,
                    in_neuron: e1,
                    out_neuron: e2,
                    disabled: false
                });
                *cen += 1;
            }
        }
        if self.links.len() > 0 && rng.gen_bool(0.2) {
            self.links.remove(rng.gen_range(0..self.links.len()));
        }
        if self.neurons.len() > IN+OUT && rng.gen_bool(0.2) {
            let n = self.neurons.remove(rng.gen_range((IN+1)..=(self.neurons.len()-OUT)));
            let mut r = 0;
            for i in 0..self.links.len() {
                let l = &self.links[i-r];
                if l.in_neuron == n.evo_num || l.out_neuron == n.evo_num {
                    self.links.remove(i-r);
                    r += 1;
                }
            }
        }
    }
    fn mix(&self, other: &Self) -> Self {
        let mut neurons = self.neurons.clone();
        for n in neurons.iter_mut() {
            if let Some(on) = other.neurons.iter().find(|on| on.evo_num == n.evo_num) {
                n.bias = (n.bias + on.bias) / 2.;
            }
        }
        let mut links = self.links.clone();
        for l in links.iter_mut() {
            if let Some(ol) = other.links.iter().find(|ol| ol.evo_num == l.evo_num) {
                l.strength = (l.strength + ol.strength) / 2.;
            }
        }
        Self {
            neurons,
            links
        }
    }
    fn draw(&self, file_name: &str) {
        use draw::*;

        let mut canvas = Canvas::new(1000, 1000);

        let mut rng = thread_rng();
        let mut neurons = HashMap::with_capacity(self.neurons.len());
        for (i,n) in self.neurons.iter().enumerate() {
            let pos = Point::new((i.min(self.neurons.len()-OUT).saturating_sub(IN-1) as f32) * 1000. / (self.neurons.len() as f32), rng.gen_range(0. ..1000.));
            canvas.display_list.add(
                Drawing::new()
                    .with_shape(Shape::Circle { radius: 4 })
                    .with_position(pos)
                    .with_style(Style::filled(Color::black()))
            );
            neurons.insert(n.evo_num, pos);
        }

        for l in self.links.iter() {
            let point1 = neurons.get(&l.in_neuron).unwrap();
            let point2 = neurons.get(&l.out_neuron).unwrap();
            canvas.display_list.add(
                Drawing::new()
                    .with_shape(LineBuilder::new(point1.x, point1.y).line_to(point2.x, point2.y).build())
                    .with_style(Style::stroked((l.strength.abs()*3.) as u32, RGB::new((-l.strength*156.) as u8, (l.strength*156.) as u8, 0)))
            );
        }

        // // create a new drawing
        // let mut rect = Drawing::new()
        //     // give it a shape
        //     .with_shape(Shape::Line { start: , points: () })
        //     // move it around
        //     .with_xy(25.0, 25.0)
        //     // give it a cool style
        //     .with_style(Style::stroked(5, Color::black()));

        // // add it to the canvas
        // canvas.display_list.add(rect);

        // save the canvas as an svg
        render::save(
            &canvas,
            file_name,
            SvgRenderer::new(),
        ).expect("Failed to save");
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
    fn train(&mut self, turnover: F) {
        let mut gen = self.current_gen.take().unwrap_or_else(|| array::from_fn(|_| (Phenotype::default(), 0.)));
        let mut random_ns = gen.choose_multiple(&mut thread_rng(), (GENERATION_SIZE as F * turnover * 2.) as usize);
        let mut children = Vec::with_capacity((GENERATION_SIZE as F * turnover) as usize);
        while let Some(((n1, s1), (n2, s2))) = random_ns.next().and_then(|x1| random_ns.next().map(|x2| (x1, x2))) {
            if s1 > s2 {
                children.push(n1.mix(n2));
            } else {
                children.push(n2.mix(n1));
            }
        }
        for (i, c) in children.into_iter().enumerate() {
            gen[gen.len()-1-i] = (c, 0.);
        }
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
    fn best(&self) -> Option<Phenotype<IN, OUT>> {
        self.current_gen.as_ref().map(|g| g[0].0.clone())
    }
}

fn main() {
    let mut trainer = Trainer::<2, 1>::new(|gen| gen.map(|nn| {
        -nn.eval([0., 0.])[0]+nn.eval([1., 0.])[0]+nn.eval([0., 1.])[0]-nn.eval([1., 1.])[0]
    }));

    let mut best_scores = Vec::new();
    for _ in 0..10000 {
        best_scores.push(trainer.best_score().unwrap_or_default());
        trainer.train(0.5);
    }
    let best = dbg!(trainer.best().unwrap());
    let nn = best.nn();
    best.draw("best.svg");
    dbg!(nn.eval([0., 0.])[0], nn.eval([1., 0.])[0], nn.eval([0., 1.])[0], nn.eval([1., 1.])[0]);

    let root_drawing_area = BitMapBackend::new("graph.png", (1024, 768))
        .into_drawing_area();

    root_drawing_area.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root_drawing_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(0..best_scores.len(), best_scores.clone().into_iter().reduce(F::min).unwrap()..best_scores.clone().into_iter().reduce(F::max).unwrap())
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart.draw_series(LineSeries::new(
        best_scores.into_iter().enumerate(),
        &RED
    )).unwrap();
}
