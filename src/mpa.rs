use rand::distributions::Distribution;
use rand::distributions::Uniform;


fn mpa (searchagents_no : usize , max_iter : usize, lb : f64, ub : f64, dim : usize, fobj : &dyn Fn(Vec<f64>)->f64) -> f64 {
     

     println!("computation with : n= {}, d= {}, kmax= {}, lb= {}, ub ={} of ",searchagents_no, dim, max_iter, lb,ub);
     
     let mut top_predator_pos = vec![0.0f64; dim]; // Top_predator_pos=zeros(1,dim);  the best solution :
     let mut top_predator_fit : f64 = 0.0f64; //f64::MAX;  // Top_predator_fit=inf;  the best fitness
     let mut convergence_curve = vec![0.0f64; max_iter]; //Convergence_curve=zeros(1,Max_iter); the best chart
     let mut stepsize = vec![vec![0.064; dim]; searchagents_no];//stepsize=zeros(SearchAgents_no,dim);
     
     let mut fitness = vec![0.0f64; searchagents_no]; //fitness=inf(SearchAgents_no,1);
     for fit in fitness.iter_mut(){
          *fit = f64::MAX;
     }

     
      let mut prey = initialization(searchagents_no, dim, lb, ub); //Prey=initialization(SearchAgents_no,dim,ub,lb);
      //write_matrix(&prey, String::from("prey"));  
      
      let xmin = repmat(searchagents_no, dim, lb); //Xmin=repmat(ones(1,dim).*lb,SearchAgents_no,1);
      let xmax = repmat(searchagents_no, dim, ub); //Xmax=repmat(ones(1,dim).*ub,SearchAgents_no,1); 
      
      write_matrix(&xmin, String::from("Xmin"));
      write_matrix(&xmax, String::from("Xmax"));

      let mut iter =0;  
      let FADs = 0.2;
      let p = 0.5;



     return top_predator_fit;   
}


fn initialization(searchagents_no : usize, dim : usize, lb : f64, ub : f64)-> Vec<Vec<f64>>{
      let mut positions = vec![vec![0.0f64; dim]; searchagents_no];
      let intervall_g = Uniform::from(0.0f64..1.0f64);
      let mut rng = rand::thread_rng();
                
      
      for i in 0..searchagents_no {
           for  j in 0..dim {   
                positions[i][j]= intervall_g.sample(&mut rng)*(ub-lb)+lb;                         
           }
      }    
      
      positions
}

fn repmat(n: usize ,d : usize, value : f64) -> Vec<Vec<f64>> {
      let mut matrix = vec![vec![0.0f64; d]; n];
          for i in 0..n {
               for j in 0..d {
                    matrix[i][j] = value;
               }
          }
      matrix
}


fn write_matrix(x: &Vec<Vec<f64>>, message :String) {

      println!("{}", message);

      for i in 0..x.len(){
           for j in 0..x[i].len() {
                print!("{} ", x[i][j]);
           }
          println!("_");
      }        
}