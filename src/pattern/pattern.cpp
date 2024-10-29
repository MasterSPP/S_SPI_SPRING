#include <vector>

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include "pattern.h"
#include <random>
#include <tuple>
#define INNER_STEPS 10


#define WFRAC  0.1
    
    
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 

std::vector<double> Pattern::downscale(std::array<std::size_t,2> newdims){
        std::vector<double> outdata(newdims[0]*newdims[1],0);
        std::vector<double> tempweight(newdims[0]*newdims[1],0);
        std::vector<double> totweight(newdims[0]*newdims[1],0);
        
        
        double scalefactor[2]={double(this->dims[0])/double(newdims[0]), double(this->dims[1])/double(newdims[1])}; 
        
        
        for(std::size_t y = 0; y<this->dims[1]; y++)
          for(std::size_t x = 0; x<this->dims[0]; x++){
              std::size_t index= x+y*this->dims[0];
              if(this->data[index]>=0){
                double ix = double(x)/scalefactor[0];
                double iy = double(y)/scalefactor[0];
                
                int ixt = std::floor(ix);
                int iyt = std::floor(iy);

                double xtdec = ix-double(ixt);
                double ytdec = iy-double(iyt);
                
                
                double val = this->data[index];
//                 if(val>=0){
                
                    if(ixt>=0 && iyt>=0 && ixt<newdims[0] && iyt<newdims[1] ){
                        double weight = (1.-xtdec)*(1.-ytdec);
                        if(val>=0) outdata[ixt+iyt*newdims[0]]+= val*weight;
                        if(val>=0) tempweight[ixt+iyt*newdims[0]]+= weight;
                        totweight[ixt+(iyt)*newdims[0]]+= weight;                        
                    }
                    if(ixt+1>=0 && iyt>=0 && ixt+1<newdims[0] && iyt<newdims[1] ){
                        double weight = (xtdec)*(1.-ytdec);
                        if(val>=0) outdata[ixt+1+iyt*newdims[0]]+= val*weight;
                        if(val>=0) tempweight[ixt+1+iyt*newdims[0]]+= weight;
                        totweight[ixt+1+(iyt)*newdims[0]]+= weight;                        

                    }
                    if(ixt>=0 && iyt+1>=0 && ixt<newdims[0] && iyt+1<newdims[1] ){
                        double weight = (1.-xtdec)*(ytdec);
                        if(val>=0) outdata[ixt+(iyt+1)*newdims[0]]+= val*weight;
                        if(val>=0) tempweight[ixt+(iyt+1)*newdims[0]]+= weight;
                        totweight[ixt+(iyt+1)*newdims[0]]+= weight;                        
                        
                    }
                    if(ixt+1>=0 && iyt+1>=0 && ixt+1<newdims[0] && iyt+1<newdims[1] ){
                        double weight = (xtdec)*(ytdec);
                        if(val>=0) outdata[ixt+1+(iyt+1)*newdims[0]]+= val*weight;
                        if(val>=0) tempweight[ixt+1+(iyt+1)*newdims[0]]+= weight;
                        totweight[ixt+1+(iyt+1)*newdims[0]]+= weight;
                    }
//                 }
              }
                
        }
        

        for(std::size_t i=0; i<outdata.size(); i++){
            if(tempweight[i]>WFRAC*totweight[i]) outdata[i]/=tempweight[i];
            else outdata[i]=-1;            
        }
                
        return outdata;
             
        
    }

    

std::vector<double> Pattern::downscale_and_correct(double R, double dx, std::array<std::size_t,2> newdims, std::vector<double> mask){
        std::vector<double> outdata(newdims[0]*newdims[1],0);
        std::vector<double> tempweight(newdims[0]*newdims[1],0);
        
        double maxdist=this->center[0];
        if(this->center[1]>maxdist) maxdist=this->center[1];
        if((this->dims[0] - this->center[0])>maxdist) maxdist=(this->dims[0] - this->center[0]);
        if((this->dims[1] - this->center[1])>maxdist) maxdist=(this->dims[1] - this->center[1]);
        
        double max_theta=std::atan2(dx*maxdist,R);
        double r_max_new = R * 2. * std::sin(max_theta/2.);
        double dx_new = r_max_new/maxdist;

        double scalefactor[2]={double(this->dims[0])/double(newdims[0]), double(this->dims[1])/double(newdims[1])}; 
        
        
        for(std::size_t y = 0; y<this->dims[1]; y++)
          for(std::size_t x = 0; x<this->dims[0]; x++){
              std::size_t index= x+y*this->dims[0];
              if(mask[index]>0.5){
                
                double relx = double(x)- double(this->center[0]);
                double rely = double(y)- double(this->center[1]);
                  
                double phi = std::atan2(rely,relx);
                double r_orig = std::sqrt(relx*relx+rely*rely)*dx;
                double theta = std::atan2(r_orig, R);
                

                
                double r_new = R * 2. * std::sin(theta/2.);
                double r_new_pixels = r_new/dx_new;
                
                double x_new_rel = r_new_pixels*std::cos(phi);
                double y_new_rel = r_new_pixels*std::sin(phi);
                
                  
                  
                double ix = (x_new_rel + double(this->center[0]))/scalefactor[0];
                double iy = (y_new_rel + double(this->center[1]))/scalefactor[1];
                
                int ixt = std::floor(ix);
                int iyt = std::floor(iy);

                double xtdec = ix-double(ixt);
                double ytdec = iy-double(iyt);
                
                
                double totweight=0;
                double val = this->data[index];                
                val *= 1./std::pow(std::cos(theta), 3);

                            
                if(ixt>=0 && iyt>=0 && ixt<newdims[0] && iyt<newdims[1] ){
                    double weight = (1.-xtdec)*(1.-ytdec);
                    outdata[ixt+iyt*newdims[0]]+= val*weight;
                    tempweight[ixt+iyt*newdims[0]]+= weight;
                    
                }
                if(ixt+1>=0 && iyt>=0 && ixt+1<newdims[0] && iyt<newdims[1] ){
                    double weight = (xtdec)*(1.-ytdec);
                    outdata[ixt+1+iyt*newdims[0]]+= val*weight;
                    tempweight[ixt+1+iyt*newdims[0]]+= weight;
                }
                if(ixt>=0 && iyt+1>=0 && ixt<newdims[0] && iyt+1<newdims[1] ){
                    double weight = (1.-xtdec)*(ytdec);
                    outdata[ixt+(iyt+1)*newdims[0]]+= val*weight;
                    tempweight[ixt+(iyt+1)*newdims[0]]+= weight;
                }
                if(ixt+1>=0 && iyt+1>=0 && ixt+1<newdims[0] && iyt+1<newdims[1] ){
                    double weight = (xtdec)*(ytdec);
                    outdata[ixt+1+(iyt+1)*newdims[0]]+= val*weight;
                    tempweight[ixt+1+(iyt+1)*newdims[0]]+= weight;
                }
                
              }
                
        }
        
        
        double maxweight = 0;
        for(std::size_t i=0; i<outdata.size(); i++) if(tempweight[i]>maxweight) maxweight = tempweight[i];
        
        
        for(std::size_t i=0; i<outdata.size(); i++){
            if(tempweight[i]>WFRAC*maxweight) outdata[i]/=tempweight[i];
            else outdata[i]=-1;            
        }
                
        return outdata;
             
        
    }
    
    
    
    
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 


    

std::tuple<std::vector<double>,std::vector<double>> Pattern::cut_downscale_and_correct(double R, double dx, std::array<std::size_t,2> cutdims, std::array<std::size_t,2> newdims, std::vector<double> mask){
        std::vector<double> outdata(newdims[0]*newdims[1],0);
        std::vector<double> outmask(newdims[0]*newdims[1],0);

        std::vector<double> tempweight(newdims[0]*newdims[1],0);
        std::vector<double> maskweight(newdims[0]*newdims[1],0);
        
        double maxdist=this->center[0];
        if(this->center[1]>maxdist) maxdist=this->center[1];
        if((this->dims[0] - this->center[0])>maxdist) maxdist=(this->dims[0] - this->center[0]);
        if((this->dims[1] - this->center[1])>maxdist) maxdist=(this->dims[1] - this->center[1]);
        
        double max_theta=std::atan2(dx*maxdist,R);
        double r_max_new = R * 2. * std::sin(max_theta/2.);
        double dx_new = r_max_new/maxdist;


        double scalefactor[2]={double(cutdims[0])/double(newdims[0]), double( cutdims[1])/double(newdims[1])}; 
        
        
        
        for(std::size_t y = 0; y<this->dims[1]; y++)
          for(std::size_t x = 0; x<this->dims[0]; x++){
              std::size_t index= x+y*this->dims[0];
                double relx = double(x)- double(this->center[0]);
                double rely = double(y)- double(this->center[1]);
                  
                double phi = std::atan2(rely,relx);
                double r_orig = std::sqrt(relx*relx+rely*rely)*dx;
                double theta = std::atan2(r_orig, R);
                
                double r_new = R * 2. * std::sin(theta/2.);
                double r_new_pixels = r_new/dx_new;
                
                double x_new_rel = r_new_pixels*std::cos(phi);
                double y_new_rel = r_new_pixels*std::sin(phi);
                  
                double ix = (x_new_rel)/scalefactor[0]  + double(newdims[0]/2);
                double iy = (y_new_rel)/scalefactor[1]  + double(newdims[1]/2);
                
                int ixt = std::floor(ix);
                int iyt = std::floor(iy);

                double xtdec = ix-double(ixt);
                double ytdec = iy-double(iyt);
                
                if(mask[index]>0.5){
                
                    double val = this->data[index];                
                    val *= 1./std::pow(std::cos(theta), 3);
                    if(ixt>=0 && iyt>=0 && ixt<newdims[0] && iyt<newdims[1] ){
                        double weight = (1.-xtdec)*(1.-ytdec);
                        outdata[ixt+iyt*newdims[0]]+= val*weight;
                        tempweight[ixt+iyt*newdims[0]]+= weight;
                        
                    }
                    if(ixt+1>=0 && iyt>=0 && ixt+1<newdims[0] && iyt<newdims[1] ){
                        double weight = (xtdec)*(1.-ytdec);
                        outdata[ixt+1+iyt*newdims[0]]+= val*weight;
                        tempweight[ixt+1+iyt*newdims[0]]+= weight;
                    }
                    if(ixt>=0 && iyt+1>=0 && ixt<newdims[0] && iyt+1<newdims[1] ){
                        double weight = (1.-xtdec)*(ytdec);
                        outdata[ixt+(iyt+1)*newdims[0]]+= val*weight;
                        tempweight[ixt+(iyt+1)*newdims[0]]+= weight;
                    }
                    if(ixt+1>=0 && iyt+1>=0 && ixt+1<newdims[0] && iyt+1<newdims[1] ){
                        double weight = (xtdec)*(ytdec);
                        outdata[ixt+1+(iyt+1)*newdims[0]]+= val*weight;
                        tempweight[ixt+1+(iyt+1)*newdims[0]]+= weight;
                    }
                }
                
                double val = mask[index];                
                
                if(ixt>=0 && iyt>=0 && ixt<newdims[0] && iyt<newdims[1] ){
                    double weight = (1.-xtdec)*(1.-ytdec);
                    outmask[ixt+iyt*newdims[0]]+= val*weight;
                    maskweight[ixt+iyt*newdims[0]]+= weight;
                    
                }
                if(ixt+1>=0 && iyt>=0 && ixt+1<newdims[0] && iyt<newdims[1] ){
                    double weight = (xtdec)*(1.-ytdec);
                    outmask[ixt+1+iyt*newdims[0]]+= val*weight;
                    maskweight[ixt+1+iyt*newdims[0]]+= weight;
                }
                if(ixt>=0 && iyt+1>=0 && ixt<newdims[0] && iyt+1<newdims[1] ){
                    double weight = (1.-xtdec)*(ytdec);
                    outmask[ixt+(iyt+1)*newdims[0]]+= val*weight;
                    maskweight[ixt+(iyt+1)*newdims[0]]+= weight;
                }
                if(ixt+1>=0 && iyt+1>=0 && ixt+1<newdims[0] && iyt+1<newdims[1] ){
                    double weight = (xtdec)*(ytdec);
                    outmask[ixt+1+(iyt+1)*newdims[0]]+= val*weight;
                    maskweight[ixt+1+(iyt+1)*newdims[0]]+= weight;
                }  
                
            
                
              }
                
//         }
        
        
        double maxweight = 0;
        for(std::size_t i=0; i<outdata.size(); i++) if(tempweight[i]>maxweight) maxweight = tempweight[i];
        
        
        for(std::size_t i=0; i<outdata.size(); i++){
            if(tempweight[i]>WFRAC*maxweight) outdata[i]/=tempweight[i];
            else outdata[i]=-1;            
        }
            
    
            
        double maxmaskweight = 0;
        for(std::size_t i=0; i<outmask.size(); i++) if(maskweight[i]>maxmaskweight) maxmaskweight = maskweight[i];
        
        
        for(std::size_t i=0; i<outmask.size(); i++){
            if(maskweight[i]>WFRAC*maxmaskweight) outmask[i]/=maskweight[i];
            else outmask[i]=-1;            
        }

        return std::make_tuple(outdata,outmask);
             
        
    }



std::tuple<std::vector<double>,std::vector<double>> Pattern::cut_and_downscale(std::array<std::size_t,2> cutdims, std::array<std::size_t,2> newdims, std::vector<double> mask){
        std::vector<double> outdata(newdims[0]*newdims[1],0);
        std::vector<double> outmask(newdims[0]*newdims[1],0);

        std::vector<double> tempweight(newdims[0]*newdims[1],0);
        std::vector<double> maskweight(newdims[0]*newdims[1],0);
        
//         std::array<double,2> scalefactor({double(this->dims[0])/double(newdims[0]), double(this->dims[1])/double(newdims[1])}); 
        double maxdist=this->center[0];
        if(this->center[1]>maxdist) maxdist=this->center[1];
        if((this->dims[0] - this->center[0])>maxdist) maxdist=(this->dims[0] - this->center[0]);
        if((this->dims[1] - this->center[1])>maxdist) maxdist=(this->dims[1] - this->center[1]);

//         double scalefactor[2]={double(this->dims[0])/double(newdims[0]), double(this->dims[1])/double(newdims[1])}; 
        double scalefactor[2]={double(cutdims[0])/double(newdims[0]), double( cutdims[1])/double(newdims[1])}; 
        
        
        
        for(std::size_t y = 0; y<this->dims[1]; y++)
          for(std::size_t x = 0; x<this->dims[0]; x++){
              std::size_t index= x+y*this->dims[0];
//               if(mask[index]>0.5){
                
                double relx = double(x)- double(this->center[0]);
                double rely = double(y)- double(this->center[1]);
                  
                 
                  
                double ix = (relx)/scalefactor[0]  + double(newdims[0]/2);
                double iy = (rely)/scalefactor[1]  + double(newdims[1]/2);
                
                int ixt = std::floor(ix);
                int iyt = std::floor(iy);

                double xtdec = ix-double(ixt);
                double ytdec = iy-double(iyt);
                
                if(mask[index]>0.5){
                
                    double val = this->data[index];                
                    if(ixt>=0 && iyt>=0 && ixt<newdims[0] && iyt<newdims[1] ){
                        double weight = (1.-xtdec)*(1.-ytdec);
                        outdata[ixt+iyt*newdims[0]]+= val*weight;
                        tempweight[ixt+iyt*newdims[0]]+= weight;
                        
                    }
                    if(ixt+1>=0 && iyt>=0 && ixt+1<newdims[0] && iyt<newdims[1] ){
                        double weight = (xtdec)*(1.-ytdec);
                        outdata[ixt+1+iyt*newdims[0]]+= val*weight;
                        tempweight[ixt+1+iyt*newdims[0]]+= weight;
                    }
                    if(ixt>=0 && iyt+1>=0 && ixt<newdims[0] && iyt+1<newdims[1] ){
                        double weight = (1.-xtdec)*(ytdec);
                        outdata[ixt+(iyt+1)*newdims[0]]+= val*weight;
                        tempweight[ixt+(iyt+1)*newdims[0]]+= weight;
                    }
                    if(ixt+1>=0 && iyt+1>=0 && ixt+1<newdims[0] && iyt+1<newdims[1] ){
                        double weight = (xtdec)*(ytdec);
                        outdata[ixt+1+(iyt+1)*newdims[0]]+= val*weight;
                        tempweight[ixt+1+(iyt+1)*newdims[0]]+= weight;
                    }
                }
                
                double val = mask[index];                
                
                if(ixt>=0 && iyt>=0 && ixt<newdims[0] && iyt<newdims[1] ){
                    double weight = (1.-xtdec)*(1.-ytdec);
                    outmask[ixt+iyt*newdims[0]]+= val*weight;
                    maskweight[ixt+iyt*newdims[0]]+= weight;
                    
                }
                if(ixt+1>=0 && iyt>=0 && ixt+1<newdims[0] && iyt<newdims[1] ){
                    double weight = (xtdec)*(1.-ytdec);
                    outmask[ixt+1+iyt*newdims[0]]+= val*weight;
                    maskweight[ixt+1+iyt*newdims[0]]+= weight;
                }
                if(ixt>=0 && iyt+1>=0 && ixt<newdims[0] && iyt+1<newdims[1] ){
                    double weight = (1.-xtdec)*(ytdec);
                    outmask[ixt+(iyt+1)*newdims[0]]+= val*weight;
                    maskweight[ixt+(iyt+1)*newdims[0]]+= weight;
                }
                if(ixt+1>=0 && iyt+1>=0 && ixt+1<newdims[0] && iyt+1<newdims[1] ){
                    double weight = (xtdec)*(ytdec);
                    outmask[ixt+1+(iyt+1)*newdims[0]]+= val*weight;
                    maskweight[ixt+1+(iyt+1)*newdims[0]]+= weight;
                }  
                
            
                
              }
                
//         }
        
        
//         double maxweight = 0;
//         for(std::size_t i=0; i<outdata.size(); i++) if(tempweight[i]>maxweight) maxweight = tempweight[i];
        
        
        for(std::size_t i=0; i<outdata.size(); i++){
            if(tempweight[i]>WFRAC*maskweight[i]) outdata[i]/=tempweight[i];
            else outdata[i]=-1;            
        }
            
    
            
        double maxmaskweight = 0;
        for(std::size_t i=0; i<outmask.size(); i++) if(maskweight[i]>maxmaskweight) maxmaskweight = maskweight[i];
        
        
        for(std::size_t i=0; i<outmask.size(); i++){
            if(maskweight[i]>WFRAC*maxmaskweight) outmask[i]/=maskweight[i];
            else outmask[i]=-1;            
        }

        return std::make_tuple(outdata,outmask);
             
        
    }
    
    

double Pattern::get_data(std::array<std::size_t,2> coords){
    std::size_t index = coords[0] + coords[1] * this->dims[0];
    if(index>=this->dims[1] * this->dims[0]) return -1;
    else return this->data[index];        
    
}


std::vector<double> Pattern::get_radial_profile(std::vector<double> angles, std::array<double,2> phi_limits){
    
    std::vector<double> profile(angles.size(), 0);
    std::vector<double> norm(angles.size(), 0);
    
    
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for(std::size_t y = 0; y<this->dims[1]; y++)
        for(std::size_t x = 0; x<this->dims[0]; x++){
            double xrel=(x-this->center[0]);
            double yrel=(y-this->center[1]);
            
            
            double angle=std::atan2(yrel, xrel)/M_PI*180.;
            while(angle+360. <= phi_limits[1]){ angle+=360.;}
            
            if(angle<=phi_limits[1] && angle>=phi_limits[0]){
                double radius = std::sqrt(xrel*xrel + yrel*yrel)*this->dtheta;
                int rad_index=-1;
//                 int ok=-1;
                
                for(int i=0; i<angles.size()-1; i++){
                    if(angles[i]<=radius && angles[i+1]>radius){
                        rad_index=i;
//                         printf("%d \n", rad_index);
                        break;                    
                    }
                }    
                
                
                if(rad_index>=0){
                    double val=this->get_data({x,y});
                    if(val>=0){
                        double factor=(angles[rad_index+1]-radius)/(angles[rad_index+1]-angles[rad_index]);
        
                        #pragma omp atomic
                        profile[rad_index] += val*factor;
                        
                        #pragma omp atomic
                        norm[rad_index] +=factor;
                        
                        #pragma omp atomic
                        profile[rad_index+1] += val*(1.-factor);
                        
                        #pragma omp atomic
                        norm[rad_index+1] +=(1.-factor);                    
                    }
                    
                    
                }

            }
            
            
        }
        
        
    
    for( std::size_t iang=0; iang<angles.size(); iang++){
            if(norm[iang]>4) profile[iang]/=norm[iang];
            else profile[iang]=-1;
        }
    
    return profile;  
    
}

