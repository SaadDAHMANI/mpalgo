use rand::distributions::Distribution;
use rand::distributions::Uniform;


fn mpa (searchagents_no : usize , max_iter : usize, lb : f64, ub : f64, dim : usize, fobj : &dyn Fn(&Vec<f64>)->f64) -> f64 {
     
      println!("computation with : n= {}, d= {}, kmax= {}, lb= {}, ub ={} of ",searchagents_no, dim, max_iter, lb,ub);
     
      let mut top_predator_pos = vec![0.0f64; dim]; // Top_predator_pos=zeros(1,dim);  the best solution :
      let mut top_predator_fit : f64 = f64::MAX;  // Top_predator_fit=inf;  the best fitness
      let mut convergence_curve = vec![0.0f64; max_iter]; //Convergence_curve=zeros(1,Max_iter); the best chart
      let mut stepsize = vec![vec![0.064; dim]; searchagents_no];//stepsize=zeros(SearchAgents_no,dim);
     
      let mut fitness = vec![0.0f64; searchagents_no]; //fitness=inf(SearchAgents_no,1);
      for fit in fitness.iter_mut(){
           *fit = f64::MAX;
      }

      let mut fit_old = vec![0.0f64; searchagents_no]; //for fitness memory
      let mut inx = vec![0.0f64; searchagents_no];     
      let mut prey = initialization(searchagents_no, dim, lb, ub); //Prey=initialization(SearchAgents_no,dim,ub,lb);
      //write_matrix(&prey, String::from("prey"));
      
      let mut prey_old = vec![vec![0.0f64; dim]; searchagents_no];
      
      let xmin = repmat(searchagents_no, dim, lb); //Xmin=repmat(ones(1,dim).*lb,SearchAgents_no,1);
      let xmax = repmat(searchagents_no, dim, ub); //Xmax=repmat(ones(1,dim).*ub,SearchAgents_no,1); 
      
      // write_matrix(&xmin, String::from("Xmin"));
      // write_matrix(&xmax, String::from("Xmax"));

      let mut iter =0;
      let mut cf: f64;  
      let FADs = 0.2;
      let p = 0.5;

      //-------------for random values generation ----------
      let intervall01 = Uniform::from(0.0f64..1.0f64);
      let mut rng = rand::thread_rng();              
      
      //----------------------------------------------------
       
      while iter < max_iter {
           //------------------- Detecting top predator ----------------- 
          for i in 0..searchagents_no {

                // space bound    
                for i in 0.. searchagents_no {
                     for j in 0..dim {
                          if prey[i][j] < lb || prey[i][j] > ub {
                               prey[i][j] = intervall01.sample(&mut rng)*(ub-lb)+lb;
                          }
                     } 
                }
               
                fitness[i]=fobj(&prey[i]); //fitness(i,1)=fobj(Prey(i,:));
           
                // if fitness(i,1)<Top_predator_fit 
                // Top_predator_fit=fitness(i,1); 
                // Top_predator_pos=Prey(i,:);
                // end
                
                if fitness[i] < top_predator_fit { //copy informations of the best search agent
                     top_predator_fit = fitness[i];
                     for j in 0..dim {
                          top_predator_pos[j]=prey[i][j];
                     } 
                }
          }  

          //------------------- Marine Memory saving -------------------
          if iter == 0 { 
               for i in 0..searchagents_no {
                fit_old[i] = fitness[i];       
               }   
            
               for i in 0..searchagents_no {
                    for j in 0..dim {
                     prey_old[i][j] = prey[i][j];
                   }    
              }
          }

          inferior(&mut inx, &fit_old, &fitness); //  Inx=(fit_old<fitness);
          //write_vector(&inx, String::from("inx"));
           
          //Prey=Indx.*Prey_old+~Indx.*Prey;
          let indx = repmat2(&inx, dim); //Indx=repmat(Inx,1,dim);
          let tild_indx = tild(&indx);

          for i in 0..searchagents_no {
               for j in 0..dim {
                    prey[i][j]=indx[i][j]*prey_old[i][j]+tild_indx[i][j]*prey[i][j];
               }
          }
          
          //fitness=Inx.*fit_old+~Inx.*fitness;  
         let tild_inx = tild2(&inx);
         for i in 0..searchagents_no {
              fitness[i]=inx[i]*fit_old[i]+tild_inx[i]*fitness[i];
         }

         //fit_old=fitness; 
          for i in 0..searchagents_no {
               fit_old[i]=fitness[i];
          }

         //Prey_old=Prey;
          for i in 0..searchagents_no {
               for j in 0..dim {
                    prey_old[i][j]=prey[i][j];
               }
          }
          //------------------------------------------------------------   
           
          let elite = repmat2(&top_predator_pos, searchagents_no);
          //write_matrix(&elite, String::from("elite"));
          let root  : f64 = (1-(iter/max_iter)) as f64;
          let powr : f64 = (2*iter/max_iter) as f64; 

          cf=root.powf(powr);

          iter +=1;  
      }

     return top_predator_fit;   
}


fn initialization(searchagents_no : usize, dim : usize, lb : f64, ub : f64)-> Vec<Vec<f64>>{
      let mut positions = vec![vec![0.0f64; dim]; searchagents_no];
      let intervall01 = Uniform::from(0.0f64..1.0f64);
      let mut rng = rand::thread_rng();              
      
      for i in 0..searchagents_no {
           for  j in 0..dim {   
                positions[i][j]= intervall01.sample(&mut rng)*(ub-lb)+lb;                         
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

fn repmat2(source : &Vec<f64>, times : usize)-> Vec<Vec<f64>> {
      let l = source.len();
      let mut outvec = vec![vec![0.0f64; times]; l];      
      for i in 0..l {
          for j in 0..times {
               outvec[i][j]= source[i];
          }  
      }
      outvec
}

fn inferior(result : &mut Vec<f64>, x : &Vec<f64>, y : &Vec<f64>) {
     let t = x.len();
     for i in 0..t {
          if x[i]<y[i] {
              result[i]=1.0f64;  
          } 
          else {result[i]=0.0f64;}
     }
}

fn tild(source : &Vec<Vec<f64>>)-> Vec<Vec<f64>> {
     let m = source.len();
     let n = source[0].len();
     let mut outmatrix = vec![vec![0.0f64; n]; m];

     for i in 0..m {
          for j in 0..n {
               if source[i][j]==1.0f64 {
                    outmatrix[i][j]=0.0f64;
               }
               else {
                    outmatrix[i][j]=1.0f64;
               }
          }
     }
     outmatrix
} 

fn tild2(source : &Vec<f64>)-> Vec<f64> {
      let m = source.len();
      let mut outmatrix = vec![0.0f64; m];

     for i in 0..m {
          if source[i]==1.0f64 {
                outmatrix[i]= 0.0f64;
          }
          else {
                outmatrix[i]=1.0f64;
          }
     }
   
     outmatrix
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


fn write_vector(x: &Vec<f64>, message :String) {

     println!("{}", message);
     for i in 0..x.len(){
               print!("{} ", x[i]);
      } 
     println!("_");       
}