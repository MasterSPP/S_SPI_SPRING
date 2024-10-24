#include "cpu/cpu_context.h"

#ifdef WITH_CUDA
#include "gpu/gpu_context.h"
#endif
#include <random>
#include "solver.h"

    Solver::Solver(FLOAT* pattern_in, std::array<std::size_t,2> dims_in, std::array<int,2> center_in,  FLOAT reality_, FLOAT realpen_, int nthreads_, bool use_bounds_, std::vector<int> GPU_ids, int seed): dims(dims_in), size(dims_in[0]*dims_in[1]), center(center_in), reality(reality_), realpen(realpen_), pattern(dims_in[0]*dims_in[1]), contexts(nthreads_){
        
        set_threads(nthreads_);
        
        for(int i=0; i<size; i++){
            pattern[i]=pattern_in[i];
        } 
        
        
        if(center[0]<0) center[0]=dims[0]/2;
        if(center[1]<0) center[1]=dims[1]/2;

        set_boundaries();
        prepare(use_bounds_);
       
        
#ifdef WITH_CUDA
        if(GPU_ids.size()==0){
            for(int ithread=0; ithread<nthreads; ithread++){
                contexts.push_back(new CPU_context(dims_in[0], (ithread+1)*seed, &pattern[0]));
            }
        }
        else{
            #pragma omp parallel num_threads(nthreads)
            {
                #pragma omp critical
                {
                int icontext = omp_get_thread_num();
                contexts[icontext] = new GPU_context(dims_in[0],(icontext+1)*seed, &pattern[0], GPU_ids[icontext%GPU_ids.size()], 32);             
                }
            }
        }
#else
        if(GPU_ids.size()>0){
            printf("Warning: No GPU support. Falling back to CPU\n");
        }
        for(int ithread=0; ithread<nthreads; ithread++)
            contexts.push_back(new CPU_context(dims_in[0], (ithread+1)*seed, &pattern[0]));

#endif

        shiftmin=0;
        shiftmax=2.;
    }
    
    
    void Solver::set_pattern(FLOAT* pattern_in){
        
       
        memcpy(&pattern[0], pattern_in, dims[0]*dims[1]*sizeof(FLOAT));

        prepare(false);

        for(auto& context : contexts){
            context->set_pattern(&pattern[0]);
        }
        
    }
    
    
    void Solver::set_shift_param(FLOAT shiftmin_, FLOAT shiftmax_){
        shiftmin=shiftmin_;
        shiftmax=shiftmax_;        
    }

    

    Solver::~Solver(){
        for(auto& context : contexts){
            delete context;
        }
        
    }
    
    
    std::function<void(Context*)> Solver::get_algorithm(std::tuple<std::string, std::vector<FLOAT>> alg){

        auto params=std::get<1>(alg);

        
        if(std::get<0>(alg)==std::string("ER") and std::get<1>(alg).size()==1){

            return [this, params]
                    (Context* context)
                    {
                    return context->ER(params[0], reality);
                                 
                    };
                    
            
        }

        else if(std::get<0>(alg)==std::string("HIO") and std::get<1>(alg).size()==2){
            return [this, params]
                    (Context* context)
                    {
                    return context->HIO(params[0], params[1], reality);
                                 
                    };
            
        }
        

        else if(std::get<0>(alg)==std::string("RAAR") and std::get<1>(alg).size()==2){
            return [this, params]
                    (Context* context)
                    {
                    return context->RAAR(params[0], params[1], reality);
                                 
                    };
            
        }
        
        else if(std::get<0>(alg)==std::string("SW") and std::get<1>(alg).size()==2){
            return [this, params]
                    (Context* context)
                    {
                    return context->SW(params[0], params[1], reality);
                                 
                    };
            
        }
        
        else if(std::get<0>(alg)==std::string("SWA") and std::get<1>(alg).size()==2){
            return [this, params]
                    (Context* context)
                    {
                    return context->SWA(params[0], params[1], reality);
                                 
                    };
            
        }
        
        
        else{
            std::cout << std::get<0>(alg) << " [ ";
            for(auto param : params) std::cout<< param << " ";
            std::cout << "] ";
            std::cout << " NOT RECOGNIZED. Skipping." ;
            return [this, params]
            (Context* context)
            {
            return;
                            
            };
            
        }
        
        
        
    }
    
    
    void Solver::set_sequence(std::vector<std::tuple<std::string, std::vector<FLOAT>>> sequence){
        
        algorithms_seq.resize(0);
        
        for(auto alg : sequence){
            algorithms_seq.push_back(get_algorithm(alg));
        }
        
    }
    
    
    void Solver::prepare(bool use_bounds){ 
        FLOAT* temp;
        temp = new FLOAT[size];

        
        for(int i=0; i<size; i++){
            temp[i]=pattern[i];
        } 


        for(int j=0; j<dims[1]; j++)
            for(int i=0; i<dims[0]; i++){
                int kx=i-center[0];
                int ky=j-center[1];
                
                if(kx<0) kx+=dims[0];
                if(ky<0) ky+=dims[1];

                int oldindex=i+dims[0]*j;
                int index=kx+dims[0]*ky;

                if(temp[oldindex]>=0)
                    pattern[index]=std::sqrt(temp[oldindex]);
                else
                    if(use_bounds)
                        pattern[index]=temp[oldindex];
                    else
                        pattern[index]=-1;
                    
        }
  
        delete [] temp;

    }
    

    void Solver::set_boundaries(){
        int binsize=1;
        int span=1;
        int profsize = int(std::sqrt(center[0]*center[0]+center[1]*center[1])/FLOAT(binsize))+10;
        std::vector<FLOAT> radprof(profsize,0);
        std::vector<FLOAT> radprofstd(profsize,0);
        std::vector<FLOAT> radmax(profsize,0);
        std::vector<FLOAT> counts(profsize,0);
        std::vector<FLOAT> totcounts(profsize,0);
        std::vector<FLOAT>   totsaturated(profsize,0);
        

        for(int j=0; j<dims[1]; j++)
            for(int i=0; i<dims[0]; i++){
                

                int index=i+dims[0]*j;
                
                FLOAT val=pattern[index];

                FLOAT dist = std::sqrt((j-center[1])*(j-center[1]) + (i-center[0])*(i-center[0]));
                int distindexcent=int(std::round(dist/FLOAT(binsize)));     
                
                for(int distindex=distindexcent-span; distindex<=distindexcent+span; distindex++){
                    if(distindex<profsize and distindex>=0){
                        if(val>=0){

                            radprof[distindex]+=val;
                            radprofstd[distindex]+=val*val;
                            if(radmax[distindex]<val) radmax[distindex]=val;
                            counts[distindex]+=1.;                        
                        }
                        if(val<-1.5) totsaturated[distindex]+=1.;
                        
                        totcounts[distindex]+=1.; 
                    }
                }

                    
        }
        
        for(int i=0; i<profsize; i++){
            if(counts[i]>totcounts[i]*0.66 and totsaturated[i]<=totcounts[i]*0.05 and totcounts[i]>40){
                radprof[i]/=counts[i];
                radprofstd[i]=std::sqrt(std::abs(radprofstd[i]/counts[i]-radprof[i]*radprof[i]));
            }
            else radprof[i]=-1;
            
        }
        
        
        

        
        for(int j=0; j<dims[1]; j++)
            for(int i=0; i<dims[0]; i++){
                

                int index=i+dims[0]*j;
                
                FLOAT val=pattern[index];
                
                if(val<0){
                    FLOAT dist = std::sqrt((j-center[1])*(j-center[1]) + (i-center[0])*(i-center[0]));
                    int distindex=int(std::round(dist/FLOAT(binsize)));     
                    
                    if(distindex<profsize and distindex>0){
                        if(radprof[distindex]>=0 and pattern[index]<-0.5 and pattern[index]>-1.5){
                            FLOAT boundval = radprof[distindex]+1.*radprofstd[distindex];

                            pattern[index]=-2.- std::sqrt(boundval);
                        }
                        else
                            pattern[index]=-1;
                    }
                    
                }
                    
        }
        
    }    
    
 std::vector<FLOAT>  Solver::get_profile(std::shared_ptr<Reconstruction> rec){ // FFT library put the origin of reciprocal space in the corners ...
        int profsize = int(std::sqrt(center[0]*center[0]+center[1]*center[1]));
        std::vector<FLOAT> radprof(profsize,0);
        std::vector<FLOAT> radprofstd(profsize,0);
        std::vector<FLOAT> counts(profsize,0);

        
        std::vector<FLOAT>  out(dims[0]*dims[1],0);
        
        

        for(int j=0; j<dims[1]; j++)
            for(int i=0; i<dims[0]; i++){
                

                int index=i+dims[0]*j;
                
                FLOAT val=pattern[index];
                
                int x,y;
                x=i;
                y=j;
                if(x>=dims[1]/2) x-=dims[1];
                if(y>=dims[1]/2) y-=dims[2];
                
                FLOAT dist = std::sqrt(x*x+y*y);
                int distindex=int(std::round(dist));     
                if(distindex<profsize){
                        radprof[distindex]+=std::abs(rec->density[index]);
                        radprofstd[distindex]+=std::abs(rec->density[index])*std::abs(rec->density[index]);
                        counts[distindex]+=1.;                        
                }
                    
        }
        
        for(int i=0; i<profsize; i++){
                radprof[i]/=counts[i];
                radprofstd[i]=std::sqrt(radprofstd[i]/counts[i]-radprof[i]*radprof[i]);
        }
        
        
        
        for(int j=0; j<dims[1]; j++)
            for(int i=0; i<dims[0]; i++){
                

                int index=i+dims[0]*j;
                               
                int x,y;
                x=i;
                y=j;
                if(x>=dims[0]/2) x-=dims[1];
                if(y>=dims[1]/2) y-=dims[2];
                
                FLOAT dist = std::sqrt(x*x+y*y);
                int distindex=int(std::round(dist));     
                if(distindex<profsize){
                        out[index]=radprof[distindex];
                }
                    
        }
        
        return out;

    }
    
// // // // // // // // // // // // // // // //     d
    
    
    std::vector<std::shared_ptr<Reconstruction>>  Solver::HIO(std::vector<std::shared_ptr<Reconstruction>> rec_array, int iterations, FLOAT beta){
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for(int irec=0; irec<rec_array.size(); irec++){
            int icontext = omp_get_thread_num();
        
            contexts[icontext]->set_density(&(rec_array[irec]->density[0]));
            contexts[icontext]->set_support(&(rec_array[irec]->support[0]));
            
            contexts[icontext]->HIO(iterations, beta, reality);

            contexts[icontext]->get_support(&(rec_array[irec]->support[0]));        
            contexts[icontext]->get_density(&(rec_array[irec]->density[0]));
            
        }
        return rec_array;

    }
    std::vector<std::shared_ptr<Reconstruction>>  Solver::RAAR(std::vector<std::shared_ptr<Reconstruction>> rec_array, int iterations, FLOAT beta){
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for(int irec=0; irec<rec_array.size(); irec++){
            int icontext = omp_get_thread_num();
        
            contexts[icontext]->set_density(&(rec_array[irec]->density[0]));
            contexts[icontext]->set_support(&(rec_array[irec]->support[0]));
            
            contexts[icontext]->RAAR(iterations, beta, reality);

            contexts[icontext]->get_support(&(rec_array[irec]->support[0]));        
            contexts[icontext]->get_density(&(rec_array[irec]->density[0]));
            
        }
        return rec_array;

    }
    std::vector<std::shared_ptr<Reconstruction>> Solver::ER(std::vector<std::shared_ptr<Reconstruction>> rec_array, int iterations){
        
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for(int irec=0; irec<rec_array.size(); irec++){
            int icontext = omp_get_thread_num();
        
            contexts[icontext]->set_density(&(rec_array[irec]->density[0]));
            contexts[icontext]->set_support(&(rec_array[irec]->support[0]));
            
            contexts[icontext]->ER(iterations, reality);

            contexts[icontext]->get_support(&(rec_array[irec]->support[0]));        
            contexts[icontext]->get_density(&(rec_array[irec]->density[0]));
            
        }
        return rec_array;

    }
    
    
     std::vector<std::shared_ptr<Reconstruction>> Solver::SW( std::vector<std::shared_ptr<Reconstruction>> rec_array, FLOAT sigma, FLOAT threshold){
        
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for(int irec=0; irec<rec_array.size(); irec++){
            int icontext = omp_get_thread_num();
        
            contexts[icontext]->set_density(&(rec_array[irec]->density[0]));
            contexts[icontext]->set_support(&(rec_array[irec]->support[0]));
            
            contexts[icontext]->SW(sigma, threshold, reality);

            contexts[icontext]->get_support(&(rec_array[irec]->support[0]));        
            contexts[icontext]->get_density(&(rec_array[irec]->density[0]));
            
        }
        return rec_array;
    }
    
    
        
     std::vector<std::shared_ptr<Reconstruction>> Solver::SWA( std::vector<std::shared_ptr<Reconstruction>> rec_array, FLOAT sigma, int  pixels){
        
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for(int irec=0; irec<rec_array.size(); irec++){
            int icontext = omp_get_thread_num();
        
            contexts[icontext]->set_density(&(rec_array[irec]->density[0]));
            contexts[icontext]->set_support(&(rec_array[irec]->support[0]));
            
            contexts[icontext]->SWA(sigma, pixels, reality);

            contexts[icontext]->get_support(&(rec_array[irec]->support[0]));        
            contexts[icontext]->get_density(&(rec_array[irec]->density[0]));
            
        }
        return rec_array;
    }
    
    
    
    std::vector<std::shared_ptr<Reconstruction>>  Solver::Sequence(std::vector<std::shared_ptr<Reconstruction>> rec_array, int nrep){
        
        
        if(algorithms_seq.size()==0){
            std::cout << "Warning: empty algorithm sequence. Operation has no effect" << std::endl;
            return rec_array;
        }
        
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for(int irec=0; irec<rec_array.size(); irec++){
            int icontext = omp_get_thread_num();
            
            contexts[icontext]->set_density(&(rec_array[irec]->density[0]));
            contexts[icontext]->set_support(&(rec_array[irec]->support[0]));
            
	    for(int irep=0;irep<nrep;irep++){
            for(auto& ALG: algorithms_seq){
                ALG(contexts[icontext]);
            }
	    }
            contexts[icontext]->get_error(reality, rec_array[irec]->error_fourier, rec_array[irec]->error_real, rec_array[irec]->complexsum, rec_array[irec]->modulesum);
            contexts[icontext]->get_symmetry(rec_array[irec]->symmetry, 1,0,1);
            
            rec_array[irec]->reality = 1.-rec_array[irec]->complexsum/rec_array[irec]->modulesum;
            
            rec_array[irec]->error =  rec_array[irec]->error_fourier*(1.+ realpen*rec_array[irec]->reality);
                    
            contexts[icontext]->get_density(&(rec_array[irec]->density[0]));
            contexts[icontext]->get_support(&(rec_array[irec]->support[0]));

        }
        
        return rec_array;
    }
    

    std::shared_ptr<Reconstruction>  Solver::Reshift_rec(std::shared_ptr<Reconstruction> rec, std::shared_ptr<Reconstruction> ref, int icontext){
        
            
        contexts[icontext]->set_density(&(rec->density[0]));
        contexts[icontext]->set_ref_density(&(ref->density[0]));
        contexts[icontext]->set_support(&(rec->support[0]));
        
        int shiftx, shifty;
        FLOAT shiftphase;
        bool flip;
        contexts[icontext]->get_shift(shiftx,shifty,shiftphase,flip, shiftmin, shiftmax);
        contexts[icontext]->reshift(shiftx,shifty,shiftphase,flip);
                
        contexts[icontext]->get_density(&(rec->density[0]));
        contexts[icontext]->get_support(&(rec->support[0]));

        return rec;
    }    

    
    
    std::vector<std::shared_ptr<Reconstruction>>  Solver::Reshift(std::vector<std::shared_ptr<Reconstruction>> rec_array, std::shared_ptr<Reconstruction> ref){
        
        
        #pragma omp parallel num_threads(nthreads)
        {
            int icontext = omp_get_thread_num();
            contexts[icontext]->set_ref_density(&(ref->density[0]));
        }

        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for(int irec=0; irec<rec_array.size(); irec++){
            int icontext = omp_get_thread_num();
            contexts[icontext]->set_density(&(rec_array[irec]->density[0]));
            contexts[icontext]->set_support(&(rec_array[irec]->support[0]));

            int shiftx, shifty;
            FLOAT shiftphase;
            bool flip;
            contexts[icontext]->get_shift(shiftx,shifty,shiftphase,flip, shiftmin, shiftmax);
            contexts[icontext]->reshift(shiftx,shifty,shiftphase,flip);

            contexts[icontext]->get_density(&(rec_array[irec]->density[0]));
            contexts[icontext]->get_support(&(rec_array[irec]->support[0]));
            
        }
        
        return rec_array;
    }
    
 
    void  Solver::Evaluate(std::vector<std::shared_ptr<Reconstruction>> rec_array){
        
        
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for(int irec=0; irec<rec_array.size(); irec++){
            int icontext = omp_get_thread_num();
            contexts[icontext]->set_density(&(rec_array[irec]->density[0]));
            contexts[icontext]->set_support(&(rec_array[irec]->support[0]));
            contexts[icontext]->get_error(reality, rec_array[irec]->error_fourier, rec_array[irec]->error_real, rec_array[irec]->complexsum, rec_array[irec]->modulesum);
            contexts[icontext]->get_symmetry(rec_array[irec]->symmetry, 1,0,1);
            
            rec_array[irec]->reality = 1.-rec_array[irec]->complexsum/rec_array[irec]->modulesum;
            
            rec_array[irec]->error =  rec_array[irec]->error_fourier*(1.+ realpen*rec_array[irec]->reality);

        }        

    }
    

    
    
    
    std::vector<FLOAT>  Solver::GetAverageModule(std::vector<std::shared_ptr<Reconstruction>> rec_array, bool ft, FLOAT fraction=1){
        
        int navg=int(fraction*FLOAT(rec_array.size()));
        if(navg>rec_array.size()) navg=rec_array.size();
        if(navg<1) navg=1;
        
        
        std::vector<int> indexes(rec_array.size());
        for(int i=0; i<rec_array.size(); i++) indexes[i]=i;
        std::sort( indexes.begin(),indexes.end(), [&](int i,int j){ return rec_array[i]->get_error()<rec_array[j]->get_error();} );

        std::vector<std::vector<FLOAT>> localavg(contexts.size(), std::vector<FLOAT>(size, 0));        
        std::vector<FLOAT> totavg(size, 0);

        
        #pragma omp parallel num_threads(nthreads)
        {
            int icontext = omp_get_thread_num();
            contexts[icontext]->set_buffer(&localavg[icontext][0]);
        }
        
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for(int irec=0; irec<navg; irec++){
            int icontext = omp_get_thread_num();
            contexts[icontext]->add_module_to_buffer(&(rec_array[indexes[irec]]->density[0]), ft);
        }
        
        
        #pragma omp parallel num_threads(nthreads)
        {
            int icontext = omp_get_thread_num();
            contexts[icontext]->get_buffer(&localavg[icontext][0]);
        }
        
        
        FLOAT* totavg_ptr = (FLOAT*) &(totavg[0]);
        #pragma omp parallel for schedule(static) num_threads(nthreads) reduction(+:totavg_ptr[:size]) 
        for(int i=0; i<size; i++) 
            for(int icontext=0; icontext<contexts.size(); icontext++){
                totavg_ptr[i]+=localavg[icontext][i];
        }
        
    
        FLOAT norm =  1./FLOAT(navg);
        
        for(int i=0; i<size; i++) totavg[i]*=norm;
        

        return totavg;

    }
    
    
    
    std::shared_ptr<Reconstruction>  Solver::GetAverage(std::vector<std::shared_ptr<Reconstruction>> rec_array, FLOAT fraction){
        int navg=int(fraction*FLOAT(rec_array.size()));
        if(navg>rec_array.size()) navg=rec_array.size();
        if(navg<1) navg=1;
                
        std::vector<int> indexes(rec_array.size());
        for(int i=0; i<rec_array.size(); i++) indexes[i]=i;
        std::sort( indexes.begin(),indexes.end(), [&](int i,int j){ return rec_array[i]->get_error()<rec_array[j]->get_error();} );

        std::vector<std::vector<SUPPORTTYPE>> localsuppavg(contexts.size(), std::vector<SUPPORTTYPE>(size, 0));
        std::vector<std::vector<COMPLEX>> localdensavg(contexts.size(), std::vector<COMPLEX>(size, 0));
        
        std::vector<SUPPORTTYPE> totsuppavg(size, 0);
        std::vector<COMPLEX> totdensavg(size, 0);

        
        #pragma omp parallel num_threads(nthreads)
        {
            int icontext = omp_get_thread_num();
            contexts[icontext]->set_buffer(&localdensavg[icontext][0]);
            contexts[icontext]->set_buffer(&localsuppavg[icontext][0]);
        }
        
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for(int irec=0; irec<navg; irec++){
            int icontext = omp_get_thread_num();
            contexts[icontext]->add_to_buffer(&(rec_array[indexes[irec]]->density[0]));
            contexts[icontext]->add_to_buffer(&(rec_array[indexes[irec]]->support[0]));
        }
        
        
        #pragma omp parallel num_threads(nthreads)
        {
            int icontext = omp_get_thread_num();
            contexts[icontext]->get_buffer(&localdensavg[icontext][0]);
            contexts[icontext]->get_buffer(&localsuppavg[icontext][0]);
        }
        
        
        #pragma omp parallel for schedule(static) num_threads(nthreads)
        for(int i=0; i<size; i++) {
            for(int icontext=0; icontext<contexts.size(); icontext++){
                totdensavg[i]+=localdensavg[icontext][i];
            }
        }

        SUPPORTTYPE* totsuppavg_ptr = (SUPPORTTYPE*) &(totsuppavg[0]);
        #pragma omp parallel for schedule(static) num_threads(nthreads) reduction(+:totsuppavg_ptr[:size]) 
        for(int i=0; i<size; i++) {
            for(int icontext=0; icontext<contexts.size(); icontext++){
                totsuppavg_ptr[i]+=localsuppavg[icontext][i];
            }
        }
        
    
        FLOAT norm =  1./FLOAT(navg);
        
        for(int i=0; i<size; i++){
            totdensavg[i]*=norm;
            if(FLOAT(totsuppavg[i])>FLOAT(navg)*0.) totsuppavg[i]=1;
            else totsuppavg[i]=0;
        }
        
        std::shared_ptr<Reconstruction> avgrec = std::make_shared<Reconstruction>(&totdensavg[0],&totsuppavg[0], dims);

        
        return avgrec;

    }
    
        
    
    std::vector<FLOAT>  Solver::get_pattern(){
        
        int npixels = contexts[0]->npixels;
        std::vector<FLOAT> pattern(npixels*npixels);
        
        contexts[0]->get_pattern(&pattern[0]);
        
        return pattern;
    }
    
    
    void Solver::set_threads(int nthreads_){
        if(nthreads_<=0){
            nthreads=omp_get_max_threads()/2;
        }
        else{
            nthreads=nthreads_;
            if(nthreads>omp_get_max_threads()){
                printf("WARNING: The number of requried threads (%d) exceeds the number of cores (%d). This usually implies performance decay\n", nthreads, omp_get_max_threads());                
            }
        }
}




std::vector<std::shared_ptr<Reconstruction>>  Solver::Crossover(std::vector<std::shared_ptr<Reconstruction>> rec_array, std::shared_ptr<Reconstruction> best_rec, std::shared_ptr<Reconstruction> midrec, FLOAT crossbalance, FLOAT crossdiff, FLOAT crossexp, FLOAT crossftin, bool crosssym, FLOAT use_average, bool update_support){
         

        std::vector<int> indexes(rec_array.size());
        for(int i=0; i<rec_array.size(); i++) indexes[i]=i;
        std::sort( indexes.begin(),indexes.end(), [&](int i,int j){ return rec_array[i]->get_error()<rec_array[j]->get_error();} );
     
    
         int bestsuppix=0;
         for(int i=0; i<size; i++) bestsuppix+=best_rec->support[i];
         FLOAT suppdim=2.*std::sqrt(FLOAT(bestsuppix)/M_PI);
    
         if(suppdim<4.) suppdim=4;
         if(suppdim>dims[0]/4.) suppdim=dims[0]/4.;
         
         auto avg_mod = GetAverageModule(rec_array, true, 0.66);

         
         #pragma omp parallel num_threads(nthreads)
        {
            int icontext = omp_get_thread_num();
            contexts[icontext]->set_avg_intensities(&(avg_mod[0]));
        }

         std::vector<std::shared_ptr<Reconstruction>> new_rec_array(rec_array.size());
         
         #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
         for(int irec=0; irec<new_rec_array.size(); irec++){

             int icontext = omp_get_thread_num();

             
             std::vector<std::shared_ptr<Reconstruction>> parents(4);
             parents[0] = std::make_shared<Reconstruction>(*rec_array[irec]);
             int extracted = int(std::pow(contexts[icontext]->rand(0, 1),crossexp)*FLOAT(rec_array.size()));
             parents[1] = std::make_shared<Reconstruction>(*rec_array[indexes[extracted]]);                
             parents[2] = std::make_shared<Reconstruction>(*rec_array[int(contexts[icontext]->rand(0, rec_array.size()-0.0001))]);
             parents[3] = std::make_shared<Reconstruction>(*rec_array[int(contexts[icontext]->rand(0, rec_array.size()-0.0001))]);
 
          
            std::shared_ptr<Reconstruction> reference = std::make_shared<Reconstruction>(*parents[0]);
 
            std::shared_ptr<Reconstruction> midreclocal = std::make_shared<Reconstruction>(*midrec);
            
            
            contexts[icontext]->set_density(&(parents[0]->density[0]));
            contexts[icontext]->set_support(&(parents[0]->support[0]));

            contexts[icontext]->set_parents_density(&(parents[1]->density[0]), &(parents[2]->density[0]), &(parents[3]->density[0]), &(midreclocal->density[0]), &(best_rec->density[0]));
            contexts[icontext]->set_parents_support(&(parents[1]->support[0]), &(parents[2]->support[0]), &(parents[3]->support[0]));
            
            
            contexts[icontext]->prepare_parents(shiftmin, shiftmax);
            
            
            bool crossft=0;
            if(contexts[icontext]->rand(0, 1)<crossftin) crossft=1;
            

            
// // 
            int squaresizex, squaresizey;

            if(crossft){
                squaresizex = int(contexts[icontext]->rand(std::ceil(FLOAT(dims[0])/suppdim), std::ceil(FLOAT(dims[0])/4.)));
                squaresizey = int(contexts[icontext]->rand(std::ceil(FLOAT(dims[0])/suppdim), std::ceil(FLOAT(dims[0])/4.))); //squaresizex; //int(contexts[icontext]->rand(1. ,std::ceil(FLOAT(dims[0])/suppdim)));
            }
            else{
                squaresizex = int(contexts[icontext]->rand(suppdim/10.+1.,suppdim/2.));
                squaresizey =  int(contexts[icontext]->rand(suppdim/10.+1.,suppdim/2.));
            }

 
            int crosssizex = (dims[0]+squaresizex-1)/squaresizex + 1;
            int crosssizey = (dims[1]+squaresizey-1)/squaresizey + 1;
            int offsetx = int(contexts[icontext]->rand(0, squaresizex));
            int offsety = int(contexts[icontext]->rand(0, squaresizey));
            FLOAT rotangle = contexts[icontext]->rand(0,M_PI);
             
            std::vector<SUPPORTTYPE> crossmap(crosssizex*crosssizey, 0);
            
            for(int i=0; i<crosssizex*crosssizey; i++){
                if(contexts[icontext]->rand(0,1)<crossbalance){
                    crossmap[i]=1;
                }
            }
            
            
            contexts[icontext]->get_new_individual(&crossmap[0], crosssizex, crosssizey, squaresizex, squaresizey, offsetx, offsety, rotangle, crossdiff, use_average, crossft,crosssym, reality);
           
            std::shared_ptr<Reconstruction> son = std::make_shared<Reconstruction>(*parents[0]);

            
            contexts[icontext]->get_density(&(son->density[0]));
            if(update_support) contexts[icontext]->get_support(&(son->support[0]));

            new_rec_array[irec]=son;
             
         }
         
         return new_rec_array;
}


// // // // // // // // // // // // // // // // // // // // // // d



    std::vector<std::shared_ptr<Reconstruction>>  Solver::Initialize_pop(int nrec, int support_size, std::array<FLOAT,2> itemsize, std::array<int,2>  nitems,  std::array<FLOAT,2> exponent, std::array<FLOAT,2> scaling_in,FLOAT phase_in,  FLOAT suppcut){
        std::vector<std::shared_ptr<Reconstruction>> rec_array(nrec);

        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for(int irec=0; irec<rec_array.size(); irec++){
            rec_array[irec] = std::make_shared<Reconstruction>(dims);
            int icontext = omp_get_thread_num();
            rec_array[irec] = Initialize_rec(rec_array[irec],  support_size,  itemsize,  nitems,   exponent, scaling_in, phase_in,  suppcut, icontext);
            
        }
        
        return rec_array;
    }
        


    std::vector<std::shared_ptr<Reconstruction>>  Solver::Initialize_pop(int nrec, std::shared_ptr<Reconstruction> refrec,  std::array<FLOAT,2> itemsize, std::array<int,2>  nitems,  std::array<FLOAT,2> exponent, std::array<FLOAT,2> scaling_in,FLOAT phase_in,  FLOAT suppcut){
        std::vector<std::shared_ptr<Reconstruction>> rec_array(nrec);

        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for(int irec=0; irec<rec_array.size(); irec++){
            rec_array[irec] = std::make_shared<Reconstruction>(dims);
            int icontext = omp_get_thread_num();
            rec_array[irec] = Initialize_rec(rec_array[irec],  refrec,  itemsize,  nitems,   exponent, scaling_in, phase_in,  suppcut, icontext);
            
        }
        
        return rec_array;
    }

    
    std::vector<std::shared_ptr<Reconstruction>>  Solver::Initialize_pop(int nrec, std::vector<SUPPORTTYPE> refsupport,  FLOAT initfrac, std::array<FLOAT,2> scaling_in, FLOAT phase_in){
        std::vector<std::shared_ptr<Reconstruction>> rec_array(nrec);

        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for(int irec=0; irec<rec_array.size(); irec++){
            rec_array[irec] = std::make_shared<Reconstruction>(dims);
            int icontext = omp_get_thread_num();
            rec_array[irec] = Initialize_rec(rec_array[irec], refsupport, initfrac, scaling_in, phase_in, icontext);
            
        }
        
        return rec_array;
    }


    std::shared_ptr<Reconstruction>  Solver::Initialize_rec(std::shared_ptr<Reconstruction> rec, int support_size,  std::array<FLOAT,2> itemsize, std::array<int,2>  nitems_in,  std::array<FLOAT,2> exponent_in, std::array<FLOAT,2> scaling_in, FLOAT phase_in, FLOAT suppcut, int icontext){
        
        int nitems = int(contexts[icontext]->rand(nitems_in[0], nitems_in[1]+0.99));
        FLOAT exponent = contexts[icontext]->rand(exponent_in[0], exponent_in[1]);
        FLOAT scaling = contexts[icontext]->rand(scaling_in[0], scaling_in[1]);
        
        
        std::vector<FLOAT> sigmas(nitems);
        std::vector<FLOAT> centerx(nitems);
        std::vector<FLOAT> centery(nitems);
        std::vector<FLOAT> phases(nitems);
        
            
        std::vector<COMPLEX> locdensity(dims[0]*dims[1]);
        std::vector<COMPLEX> locdensity_ft(dims[0]*dims[1]);
        std::vector<SUPPORTTYPE> locsupport(dims[0]*dims[1]);

        
        for(int i=0; i<  dims[0]*dims[1]; i++){
            locdensity[i]=0;
            locsupport[i]=0;
        }     

                
        for(int i=0; i<nitems; i++){
            FLOAT sigma = std::sqrt(contexts[icontext]->rand(std::pow(itemsize[0],2), std::pow(itemsize[1],2)))/2.;

            FLOAT cx = contexts[icontext]->rand(sigma, support_size-sigma);
            FLOAT cy = contexts[icontext]->rand(sigma, support_size-sigma);

            sigmas[i] = sigma;
            centerx[i] = cx;
            centery[i] = cy;
            phases[i]  = contexts[icontext]->rand(-phase_in*M_PI, phase_in*M_PI);
        }
        
        
        
        
        for(int iy=0; iy<support_size; iy++)
            for(int ix=0; ix<support_size; ix++){
                
                int index = (iy+dims[1]/2-support_size/2)*dims[0]  + (ix+dims[0]/2-support_size/2);
                locsupport[index]=1;
                for(int i=0; i<nitems; i++){
                    FLOAT rad =  std::sqrt((ix - centerx[i])*(ix - centerx[i]) + (iy - centery[i])*(iy - centery[i]));
                    FLOAT val=0;
                    if(rad<sigmas[i]){
                        val = 2.* sigmas[i]* std::sin(std::acos(rad/sigmas[i]));

                        locdensity[index]+=std::polar(val,phases[i]);
                    }
                }              
             
                FLOAT mod=std::abs(locdensity[index]);
                if(mod>0){
                    locdensity[index] = locdensity[index]/mod*std::pow(mod,exponent);
                                        
                }
            }
            
        contexts[icontext]->set_density(&(locdensity[0]));
        contexts[icontext]->get_ft();
        contexts[icontext]->get_density(&(locdensity_ft[0]));  
             
        FLOAT norm=0;
        FLOAT norm_new=0;
        
        for(int i=0; i<  dims[0]*dims[1]; i++){
            if(pattern[i]>=0){
                norm += std::pow(std::abs(locdensity_ft[i]),2);
                norm_new += pattern[i]*pattern[i];               
            }
        }
        
        for(int i=0; i<  dims[0]*dims[1]; i++){
            locdensity[i]*=std::sqrt(norm_new/norm)*scaling;
        }
               
        rec->set_density(&(locdensity[0]));
        rec->set_support(&(locsupport[0]));

        return rec;
    }    
    
    
    
    std::shared_ptr<Reconstruction>  Solver::Initialize_rec(std::shared_ptr<Reconstruction> rec, std::shared_ptr<Reconstruction> refrec, std::array<FLOAT,2> itemsize, std::array<int,2>  nitems_in,  std::array<FLOAT,2> exponent_in, std::array<FLOAT,2> scaling_in, FLOAT phase_in, FLOAT suppcut, int icontext){
        
        int nitems = int(contexts[icontext]->rand(nitems_in[0], nitems_in[1]+0.99));
        FLOAT exponent = contexts[icontext]->rand(exponent_in[0], exponent_in[1]);
        FLOAT scaling = contexts[icontext]->rand(scaling_in[0], scaling_in[1]);
        
        
        std::vector<FLOAT> sigmas(nitems);
        std::vector<FLOAT> centerx(nitems);
        std::vector<FLOAT> centery(nitems);
//         std::vector<FLOAT> scalings(nitems);
        std::vector<FLOAT> phases(nitems);
        
            
        std::vector<COMPLEX> locdensity(dims[0]*dims[1]);
//         std::vector<COMPLEX> locdensity_ft(dims[0]*dims[1]);
        std::vector<SUPPORTTYPE> locsupport = refrec->get_support();
        std::vector<COMPLEX> refdensity = refrec->get_density();
               

        int supppix=0;
        COMPLEX sum=0;
        for(int i=0; i<  dims[0]*dims[1]; i++){
            locdensity[i]=0;
            sum+=refdensity[i];
            if(locsupport[i] > 0) supppix+=1;
        }
        
        FLOAT newnorm = std::abs(sum);

                
        for(int i=0; i<nitems; i++){
//             FLOAT sigma = contexts[icontext]->rand(minsize, maxsize)/2.;
            FLOAT sigma = std::sqrt(contexts[icontext]->rand(std::pow(itemsize[0],2), std::pow(itemsize[1],2)))/2.;

            int suppindex = contexts[icontext]->rand(0, supppix);
            int suppcounter=0;
            int index = 0;
            for(int j=0; j<  dims[0]*dims[1]; j++){
                if(suppcounter==suppindex){
                    index=j;
                    break;
                }
                if(locsupport[j] > 0) suppcounter+=1;
            }   
            
            FLOAT cx = index%dims[0];
            FLOAT cy = index/dims[0];
            
            sigmas[i] = sigma;
            centerx[i] = cx;
            centery[i] = cy;
            phases[i]  = contexts[icontext]->rand(-phase_in*M_PI, phase_in*M_PI);
        }
        
        
        
        
        for(int iy=0; iy<dims[1]; iy++)
            for(int ix=0; ix<dims[0]; ix++){
                
                int index = iy*dims[0]  + ix;
                if(locsupport[index]>0){
                    for(int i=0; i<nitems; i++){            
                        FLOAT rad =  std::sqrt((ix - centerx[i])*(ix - centerx[i]) + (iy - centery[i])*(iy - centery[i]));
                        FLOAT val=0;
                        if(rad<sigmas[i]){
                            val = 2.* sigmas[i]* std::sin(std::acos(rad/sigmas[i]));
                            locdensity[index]+=std::polar(val,phases[i]);
                        }
                    }              
                
                    FLOAT mod=std::abs(locdensity[index]);
                    if(mod>0){
                        locdensity[index] = locdensity[index]/mod*std::pow(mod,exponent);
                                            
                    }
                }
                
            }
            

        COMPLEX oldsum=0;        
        for(int i=0; i<  dims[0]*dims[1]; i++){
                oldsum += locdensity[i];
        }
        FLOAT oldnorm = std::abs(oldsum);

        for(int i=0; i<  dims[0]*dims[1]; i++){
            locdensity[i]*=newnorm/oldnorm*scaling;
        }
        
               
        rec->set_density(&(locdensity[0]));
        rec->set_support(&(locsupport[0]));

        return rec;
    }
        

    std::shared_ptr<Reconstruction>  Solver::Initialize_rec(std::shared_ptr<Reconstruction> rec, std::vector<SUPPORTTYPE> support, FLOAT initfrac,  std::array<FLOAT,2> scaling_in, FLOAT phase_in, int icontext){

        const COMPLEX imag(0.0, 1.0);

        FLOAT normfactor = contexts[icontext]->rand(scaling_in[0] ,scaling_in[1]);

        int refsuppix=0;
        for(int i=0; i<size; i++) refsuppix+=support[i];
        FLOAT suppdim=2.*std::sqrt(FLOAT(refsuppix)/M_PI);

        if(suppdim<4.) suppdim=4;
        if(suppdim>dims[0]/4.) suppdim=dims[0]/4.;
    
        int squaresizex, squaresizey;
        bool crossft = true;
        if(crossft){
            squaresizex = int(contexts[icontext]->rand(1 ,FLOAT(dims[0])/16.));
            squaresizey = int(contexts[icontext]->rand(1 ,FLOAT(dims[0])/16.));
            
        }
        else{
            squaresizex = int(contexts[icontext]->rand(suppdim/10.+1.,suppdim/2.));
            squaresizey =  int(contexts[icontext]->rand(suppdim/10.+1.,suppdim/2.));
        }


        int crosssizex = (dims[0]+squaresizex-1)/squaresizex + 1;
        int crosssizey = (dims[1]+squaresizey-1)/squaresizey + 1;
        int offsetx = int(contexts[icontext]->rand(0, squaresizex));
        int offsety = int(contexts[icontext]->rand(0, squaresizey));
        FLOAT rotangle = contexts[icontext]->rand(0,M_PI);
            
        std::vector<SUPPORTTYPE> crossmap(crosssizex*crosssizey, 0);
        std::vector<FLOAT> scaling(crosssizex*crosssizey, 0);
        std::vector<FLOAT> phase(crosssizex*crosssizey, 0);
        
        for(int i=0; i<crosssizex*crosssizey; i++){
            if(contexts[icontext]->rand(0,1)<initfrac){
                crossmap[i]=1;
                scaling[i] = contexts[icontext]->rand(0.5 ,2);
                phase[i] = contexts[icontext]->rand(-M_PI*phase_in ,M_PI*phase_in); 
            }
        }
        

        std::vector<COMPLEX> locdensity(dims[0]*dims[1]);
        std::vector<COMPLEX> locdensity_ft(dims[0]*dims[1]);

        for(int i=0; i<dims[0]*dims[1]; i++) locdensity[i] = support[i];
        
        std::vector<SUPPORTTYPE> locsupport(support);
        
        contexts[icontext]->set_density(&(locdensity[0]));
        contexts[icontext]->get_ft();
        contexts[icontext]->get_density(&(locdensity[0]));          

        for(int i=0; i<dims[0]*dims[1]; i++){           
            int x = i%dims[0];
            int y = i/dims[0];
            
            if(x>=dims[0]/2) x-=dims[0];
            if(y>=dims[1]/2) y-=dims[1];
            
            FLOAT radius = std::sqrt(FLOAT(x*x+y*y)); 
            if(radius>= dims[0]/2.) radius = dims[0]/2-1;
            FLOAT angle = std::atan2(FLOAT(y),FLOAT(x));
            
            angle+=rotangle;
            x = int(std::cos(angle)*radius);
            y = int(std::sin(angle)*radius);

            x+=dims[0]/2;
            y+=dims[1]/2;   
            int crossx= (x+offsetx)/squaresizex;
            int crossy= (y+offsety)/squaresizey;
            int crossindex = crossx+crossy*crosssizex;
      
            if(crossmap[crossindex]==1){
                locdensity[i]=locdensity[i]*scaling[crossindex]*std::exp(imag * phase[crossindex]);
            }
        }        
        
        
        contexts[icontext]->set_density(&(locdensity[0]));
        contexts[icontext]->get_ift();
        contexts[icontext]->get_density(&(locdensity[0]));  
        
        
        contexts[icontext]->set_density(&(locdensity[0]));
        contexts[icontext]->get_ft();
        contexts[icontext]->get_density(&(locdensity_ft[0]));  
        
        
        FLOAT norm=0;
        FLOAT norm_new=0;
        
        for(int i=0; i<  dims[0]*dims[1]; i++){
            if(pattern[i]>=0){
                norm += std::pow(std::abs(locdensity_ft[i]),2);
                norm_new += pattern[i]*pattern[i];               
            }
        }
        
        for(int i=0; i<  dims[0]*dims[1]; i++){
            locdensity[i]*=std::sqrt(norm_new/norm)*normfactor;
        }
            
        
        
        rec->set_density(&(locdensity[0]));
        rec->set_support(&(locsupport[0]));
    
        return rec;
    }    
    

