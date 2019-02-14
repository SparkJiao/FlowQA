### baseline  

76.680

### version0  

matrix dot-product attention replaces flow3   
stopped    

### version1

matrix dot-product attention replaces flow1, flow2, and flow3  
72.621   

### version 2

dot-product attention iterates twice  
72.487  

### version3  

change number of flows from two to three  
72.972  

### version4

matrix dot-product attention with c_t-1 and c_t and then concat and project them to original hidden size  
73.211  

### version5  

use fusion layer of matrix attention and original c_t

### verison6

matrix attention + self attention   
72.467  

### version7

matrix attention + input gate

### version8  

matrix attention + two input gates

### version9

version8 with three times iteration  
76.433  

### version10

add two gates of version8  
75.860  

### version11

add two gates simple  
75.949

### version12

just concat  
75.545  

### version13 

another gate    
74.868  

### version4

add bidaf module   
76.760  