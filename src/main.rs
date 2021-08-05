//  Marine Predators Algorithm
//  paper:
//  A. Faramarzi, M. Heidarinejad, S. Mirjalili, A.H. Gandomi, 
//  Marine Predators Algorithm: A Nature-inspired Metaheuristic
//  Expert Systems with Applications
//  DOI: doi.org/10.1016/j.eswa.2020.113377
//  Original Matlab code : https://github.com/afshinfaramarzi/Marine-Predators-Algorithm

//  Re-implemented in Rust by SaadDAHMANI <sd.dahmani2000@gmail.com; s.dahmani@univ-bouira.dz>

include!("mpa.rs");

fn main() {
    println!("Marine Predators Algorithm (MPA)");
     let n : usize =4; //search agents number
     let d : usize = 5; //search space dimension 
     let kmax : usize = 1; //iterations count
     let lb : f64 =-100.00; //lower bound of the search space
     let ub : f64 = 100.00; //uper bound of the search space 

    
     let bestfit = mpa(n,kmax,lb,ub,d, &f1);

     println!("the best fitness = {}", bestfit);
     
    normal_dist();
     

}

fn f1(x : &Vec<f64>)-> f64 {
    let mut sum : f64 = 0.0;

    for value in x.iter(){
    sum += value.powi(2);
    }  
    sum
}

fn normal_dist(){
 
    // mean 2, standard deviation 3
    let normal = Normal::new(0.0, 1.16).unwrap();
    let v = normal.sample(&mut rand::thread_rng());
    println!("{} is from a N(2, 9) distribution", v);
    
   
}