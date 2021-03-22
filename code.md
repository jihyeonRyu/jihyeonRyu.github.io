---
layout: post    
title: Code Collection    
tags: [function]    
comments: false  
--- 

### Saved Model To Buffer *.txt File
```c++
void to_buffer(const char* path){
    FILE *fin = fopen(path, "rb");
    FILE *fout = fopen("out_model.txt", "wt");
    unsigned char input_buff;
    int read_cnt = 0;
    while(fread(&input_buff, sizeof(char), 1, fin)){
        fprintf(fout,"0x%02x, ", input_buff);

        read_cnt++;
        if(read_cnt % 10 == 0)
            fprintf(fout,"\n");
    }
    fclose(fin);
    fclose(fout);
}
```