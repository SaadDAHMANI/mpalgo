use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand_distr::Normal;


use rand::seq::SliceRandom;

//use rand_distr::{Normal, NormalError};
use special::Gamma;

pub const PI: f64 = 3.14159265358979323846264338327950288f64;

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
      let fads = 0.2;
      let p = 0.5;

      //-------------for random values generation ----------
      let intervall01 = Uniform::from(0.0f64..1.0f64);
      let mut rng = rand::thread_rng(); 
      
      //-------------- to use f64 format
      let mut iterf64 : f64 = 0.0f64;
      let max_iterf64 : f64 = max_iter as f64;
      
      //----------------------------------------------------
       
      while iter < max_iter {
           //------------------- Detecting top predator ----------------- 
          for i in 0..searchagents_no {

                // space bound    
                for i in 0.. searchagents_no {
                     for j in 0..dim {
                          //if prey[i][j] < lb || prey[i][j] > ub {
                            //   prey[i][j] = intervall01.sample(&mut rng)*(ub-lb)+lb;
                          //}
                          if prey[i][j] < lb {
                               prey[i][j]=lb;
                          }
                          if prey[i][j]>ub {
                               prey[i][j]=ub;
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
           
          let elite = repmat2(&top_predator_pos, searchagents_no); //%(Eq. 10)
          write_matrix(&elite, String::from("elite").as_str());

          cf=(1.0-(iterf64/max_iterf64)).powf(2.0*iterf64/max_iterf64);

          //Levy random number vector
          //RL=0.05*levy(SearchAgents_no,dim,1.5); 
          //0.05*levy(..) is integrated in levy function
          let rl = levy(searchagents_no, dim, 1.50);
          //write_matrix(&rl, &String::from("RL").as_str());
          
          //Brownian random number vector (matrix)
          //RB=randn(SearchAgents_no,dim); 
          let rb = randn(searchagents_no, dim);
          
          for i in 0..searchagents_no {

               for j in 0..dim {
                    let r = intervall01.sample(&mut rng);

                    //------------------ Phase 1 (Eq.12) -----

                    if iter < (max_iter/3) {
                          //stepsize(i,j)=RB(i,j)*(Elite(i,j)-RB(i,j)*Prey(i,j));
                          stepsize[i][j]=rb[i][j]*elite[i][j]-rb[i][j]*prey[i][j];
                    
                          // Prey(i,j)=Prey(i,j)+P*R*stepsize(i,j);
                          prey[i][j]=prey[i][j]+p*r*stepsize[i][j];
                    } 
                    //--------------- Phase 2 (Eqs. 13 & 14)------
                    else if iter>(max_iter/3) && iter<(2*max_iter/3) {
                          if i > (searchagents_no/2) {
                               //stepsize(i,j)=RB(i,j)*(RB(i,j)*Elite(i,j)-Prey(i,j));
                               stepsize[i][j]=rb[i][j]*(rb[i][j]*elite[i][j]-prey[i][j]);

                               //Prey(i,j)=Elite(i,j)+P*CF*stepsize(i,j); 
                               prey[i][j]=elite[i][j]+p*cf*stepsize[i][j];     
                          } 
                          else {
                               //stepsize(i,j)=RL(i,j)*(Elite(i,j)-RL(i,j)*Prey(i,j));
                               stepsize[i][j]=rl[i][j]*(elite[i][j]-rl[i][j]*prey[i][j]);

                               //Prey(i,j)=Prey(i,j)+P*R*stepsize(i,j); 
                               prey[i][j]=prey[i][j]+p*r*stepsize[i][j]; 
                          }
                    }
                    //----------------- Phase 3 (Eq. 15)-------
                    else {
                         //stepsize(i,j)=RL(i,j)*(RL(i,j)*Elite(i,j)-Prey(i,j)); 
                         stepsize[i][j]=rl[i][j]*(rl[i][j]*elite[i][j]-prey[i][j]); 
                         
                         //Prey(i,j)=Elite(i,j)+P*CF*stepsize(i,j);  
                         prey[i][j]=elite[i][j]+p*cf*stepsize[i][j]; 
                    }
               }
          }

          //----------------- Detecting top predator ----
          for i in 0.. searchagents_no {

               // space bound    
                for j in 0..dim {
                     //if prey[i][j] < lb || prey[i][j] > ub {
                     //   prey[i][j] = intervall01.sample(&mut rng)*(ub-lb)+lb;
                     //}
                     if prey[i][j] < lb {
                          prey[i][j]=lb;
                     }
                     if prey[i][j]>ub {
                          prey[i][j]=ub;
                     }
                }

                //fitness(i,1)=fobj(Prey(i,:));
                fitness[i]=fobj(&prey[i]);

                if fitness[i] < top_predator_fit {
                    //Top_predator_fit=fitness(i,1);
                    top_predator_fit = fitness[i];
                    
                    //Top_predator_pos=Prey(i,:); //copy information
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

     //---------- Eddy formation and FADs� effect (Eq 16) ----------- 
     
     let r = intervall01.sample(&mut rng);
     if r<fads {
           // U=rand(SearchAgents_no,dim)<FADs;
           let randmatrix = random(searchagents_no, dim);
           let u = get_u_matrix(&randmatrix,fads);
           
           //Prey=Prey+CF*((Xmin+rand(SearchAgents_no,dim).*(Xmax-Xmin)).*U);
           let randmatrix2 = random(searchagents_no, dim);   
           
           for i in 0..searchagents_no {
                for j in 0..dim {
                     prey[i][j]=prey[i][j]+cf*((xmin[i][j] + randmatrix2[i][j]*(xmax[i][j]-xmin[i][j]))*u[i][j]);
                }
           }
     }
     else {
          let rr = intervall01.sample(&mut rng);

     }






          iter +=1;
          iterf64+=1.0f64;
            
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
      let mut outvec = vec![vec![0.0f64; l]; times];      
      for i in 0..times {
          for j in 0..l {
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

fn levy(n : usize, m : usize, beta : f64)-> Vec<Vec<f64>> {

      //num = gamma(1+beta)*sin(pi*beta/2); % used for Numerator 
      let num = Gamma::gamma(1.0+beta)*(0.5*PI*beta).sin(); 
      // println!("gamma num : {}", num);
       
      // den = gamma((1+beta)/2)*beta*2^((beta-1)/2); % used for Denominator
      let den = Gamma::gamma(0.5*(1.0+beta))*beta*2.0f64.powf(0.5*(beta - 1.0)); 
      //println!("gamma den : {}",den);

      // sigma_u = (num/den)^(1/beta);% Standard deviation
      let sigma_u = (num/den).powf(1.0/beta);

      //u = random('Normal',0,sigma_u,n,m);    
      //v = random('Normal',0,1,n,m);
      //z =u./(abs(v).^(1/beta));

      let mut z = vec![vec![0.0f64; m];n];
      let mut u : f64 = 0.0;
      let mut v : f64 = 0.0;

      
      for i in 0..n {
            for j in 0..m {
                //u = random('Normal',0,sigma_u,n,m); 
                let normal_u = Normal::new(0.0, sigma_u).unwrap();
                u = normal_u.sample(&mut rand::thread_rng());

                 //v = random('Normal',0,1,n,m);    
                let normal_v =Normal::new(0.0, 1.0).unwrap();
                v = normal_v.sample(&mut rand::thread_rng());
                
                //z =u./(abs(v).^(1/beta));
                z[i][j]= (0.05*u)/(v.abs().powf(1.0/beta));
               }
            }
          return  z; 
}

fn randn(n : usize, m : usize)->Vec<Vec<f64>> {
    //Brownian random number matrix
    let mut brownian = vec![vec![0.0f64; m]; n];

    for i in 0..n {
         for j in 0..m {
          let normal =Normal::new(0.0, 1.0).unwrap();
          brownian[i][j]=normal.sample(&mut rand::thread_rng());
      }
    }
    brownian 
}

fn random(n : usize, m : usize)->Vec<Vec<f64>> {
      let mut randmatrix = vec![vec![0.0f64; m]; n];
      let intervall01 = Uniform::from(0.0f64..1.0f64);
      let mut rng = rand::thread_rng();              
      
      for i in 0..n {
           for  j in 0..m {   
                randmatrix[i][j] = intervall01.sample(&mut rng);                         
           }
      }   
     return randmatrix;
}

fn get_u_matrix (source : &Vec<Vec<f64>>, value : f64)-> Vec<Vec<f64>> {

     let n =  source.len();
     let m = source[0].len();
     let mut result = vec![vec![0.0f64; m];n];

     for i in 0..n {
          for j in 0..m {
               if source[i][j] < value {
                    result[i][j] =1.0; 
               }
               else {
                    result[i][j] =0.0;
               }
          }
     }
     return result;
} 

fn randperm (length : usize)-> Vec<usize> {

      let mut vec: Vec<usize> = (0..length).collect();
      vec.shuffle(&mut rand::thread_rng());
      return vec;
}



fn write_matrix(x: &Vec<Vec<f64>>, message :&str) {

      println!("{}", message);

      for i in 0..x.len(){
           for j in 0..x[i].len() {
                print!("{} ", x[i][j]);
           }
          println!("_");
      }        
}


fn write_vector(x: &Vec<f64>, message : &str) {

     println!("{}", message);
     for i in 0..x.len(){
               print!("{} ", x[i]);
      } 
     println!("_");       
}