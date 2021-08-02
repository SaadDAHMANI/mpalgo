

fn mpa (searchagents_no : usize , max_iter : usize, lb : f64, ub : f64, dim : usize, fobj : &dyn Fn(Vec<f64>)->f64) -> f64 {
        
     let mut top_predator_pos = vec![0.0f64; dim]; // Top_predator_pos=zeros(1,dim);  the best solution :
     let mut top_predator_fit : f64 = f64::MAX;  // Top_predator_fit=inf;  the best fitness
     let mut convergence_curve = vec![0.0f64; max_iter]; //Convergence_curve=zeros(1,Max_iter); the best chart
     let mut stepsize = vec![vec![0.064; dim]; searchagents_no];//stepsize=zeros(SearchAgents_no,dim);

     println!("computation with : n= {}, d= {}, kmax= {}, lb= {}, ub ={} of ",searchagents_no, dim, max_iter, lb,ub);



     return top_predator_fit;   
}