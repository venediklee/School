#include<stdio.h> 
#include<unistd.h> 
#include<fcntl.h> 
#include<stdlib.h> 
#include <string.h>

int main(int argc, char *argv[])
{
    int numMappers=atoi(argv[1]);//argv[1].toInt

    char *mapper_proc=argv[2];//argv[2]

    int childID=-1;//id of process,-1 for parent, 0-N for childs
    

	int fd[numMappers][2];//fd[][0]->read, fd[][1]->write ends respectively

	char parameter[32];//used for passing childID as argument

    if(argc==3)//establish map model
    {
		pid_t pid[numMappers];//pid's of childs

        // • Create necessary pipes (system call: pipe),

        
        
        for(int i = 0; i < numMappers; i++)
        {
            if(pipe(fd[i])<0)//create pipe
            {
                exit(1);
            }
        }
        
        // • Create child processes (system call: fork ),
        
        //create N childs by forking
        for(int i = 0; i < numMappers; i++)
        {
            pid[i]=fork();
            if(pid[i]==0)//child process
            {
                childID=i;
                break;//childs dont create any more childs
            }
            //else continue;
        }

        
        if(childID!=-1)//for childs
        {
            // • Duplicate corresponding pipes to the specified file descriptors of the child processes and close unused end of pipes (system call: dup2, close)
            dup2(fd[childID][0], STDIN_FILENO);//duplicate read end to std in end
            for(int i = 0; i < numMappers; i++)
            {
                //close both read & write ends 
                close(fd[i][0]);
                close(fd[i][1]);
            }
            
            // • Execute mappers and reducers (if necessary) (system call: exec family)
			sprintf(parameter, "%d", childID);
			execl(mapper_proc, mapper_proc, parameter,NULL);

        }
        else//for parent
        {
            // • Duplicate corresponding pipes to the specified file descriptors of the child processes and close unused end of pipes (system call: dup2, close),

            for(int i = 0; i < numMappers; i++)
            {
                close(fd[i][0]);//read end
            }
            

            // • Feed mappers with the given inputs (system calls: read, write),
            char str[1500];//used for reading input
			fprintf(stdout,"I am parent\n");
			fflush(stdout);
			for (int i = 0; fgets(str, 1500, stdin); i++)//while we haven't reached EOF of std input
			{
				fprintf(stdout,"str is: %s, length is %ld\n", str, strlen(str));
				fflush(stdout);
				i %= numMappers;
				//dup2(fd[i][0],STDIN_FILENO);
				write(fd[i][1], str, strlen(str));
				//write(fd[i][1], "\n", 1);
				//close(fd[i][0]);
			}
            for(int i = 0; i < numMappers; i++)
            {
                close(fd[i][1]);//write end  
            }
            
            // • Wait until all child processes terminate and then terminate itself (system call: wait family).
            for(int i = 0; i < numMappers; i++)
            {
				pid[i] = wait(NULL);
            }
            exit(EXIT_SUCCESS);
            
        
        }
        
    }
    else if(argc==4)//establish mapreduce model
    {
		char *reducer_proc = argv[3];

		pid_t pid[2*numMappers];//pid's of childs

        int fd2[numMappers][2];//used for pipe's between mapper-reducer, 
        int fd3[numMappers][2];//used for pipe's between reducer-reducer(last pipe for reducer->tty)
        // • Create necessary pipes (system call: pipe),

        for(int i = 0; i < numMappers; i++)
        {
            if(pipe(fd[i])<0)//create pipe
            {
                exit(1);
            }
            if(pipe(fd2[i])<0) exit(1);
            if(pipe(fd3[i])<0) exit(1);
        }

        // • Create child processes (system call: fork ),
        
        //create 2*N childs by forking
        for(int i = 0; i < 2*numMappers; i++)
        {
            pid[i]=fork();
            if(pid[i]==0)//child process
            {
                childID=i;
                break;//childs dont create any more childs
            }
            //else continue;
        }


        if(childID!=-1 && childID<numMappers)//for childs(mappers)
        {
            // • Duplicate corresponding pipes to the specified file descriptors of the child processes and close unused end of pipes (system call: dup2, close),
            dup2(fd[childID][0], STDIN_FILENO);//duplicate read end to std in end
			dup2(fd2[childID][1], STDOUT_FILENO);
			for(int i = 0; i < numMappers; i++)
            {
                //close both read & write ends 
                close(fd[i][0]);
                close(fd[i][1]);
                close(fd2[i][0]);
                close(fd2[i][1]);
                close(fd3[i][0]);
                close(fd3[i][1]);
                
            }
            
            // • Execute mappers and reducers (if necessary) (system call: exec family)
			sprintf(parameter, "%d", childID);
			execl(mapper_proc, mapper_proc, parameter,NULL);

        }
        else if(childID!=-1)//for childs(reducers)
        {
            // • Duplicate corresponding pipes to the specified file descriptors of the child processes and close unused end of pipes (system call: dup2, close),
            dup2(fd2[childID-numMappers][0], STDIN_FILENO);//duplicate read end to fd write end
            if(childID!=2*numMappers-1)//only !last reducer
            {
                dup2(fd3[childID-numMappers][1],STDOUT_FILENO);//duplicate write end of last child to std out end
            }
            if(childID!=numMappers) //only !first reducer
				dup2(fd3[childID-numMappers-1][0],STDERR_FILENO);//duplicate error end of next child to write end
            
            for(int i = 0; i < numMappers; i++)
            {
                //close both read & write ends 
                close(fd[i][0]);
                close(fd[i][1]);
                close(fd2[i][0]);
                close(fd2[i][1]);
                close(fd3[i][0]);
                close(fd3[i][1]);
            }
            
            // • Execute mappers and reducers (if necessary) (system call: exec family)
			sprintf(parameter, "%d", childID-numMappers);
			execl(reducer_proc, reducer_proc, parameter,NULL);
        }
        else//for parent
        {
            // • Duplicate corresponding pipes to the specified file descriptors of the child processes and close unused end of pipes (system call: dup2, close),
            for(int i = 0; i < numMappers; i++)
            {
                //dup2(fd[i][1],STDOUT_FILENO);//duplicate write end to std out end
            }
            for(int i = 0; i < numMappers; i++)
            {
                close(fd[i][0]);//read end
                close(fd2[i][0]);
                close(fd2[i][1]);
                close(fd3[i][0]);
                close(fd3[i][1]);
            }

            // • Feed mappers with the given inputs (system calls: read, write),
			char str[1500];//used for reading input

			for (int i = 0; fgets(str, 1500, stdin); i++)//while we haven't reached EOF of std input
			{
				fprintf(stdout, "str is: %s\n", str);
				fflush(stdout);
				i %= numMappers;
				//dup2(fd[i][0],STDIN_FILENO);
				write(fd[i][1], str, 1500);
				//write(fd[i][1], "\n", 1);
				//close(fd[i][0]);
			}
            for(int i = 0; i < numMappers; i++)
            {
                close(fd[i][1]);//write end  
            }
            
            // • Wait until all child processes terminate and then terminate itself (system call: wait family).
            for(int i = 0; i < 2*numMappers; i++)
            {
				 pid[i]= wait(NULL);
            }
            exit(EXIT_SUCCESS);
            
        }

    }
    else//wrong argument count
    {
        printf("wrong argc");
        exit(1);//failure
    }
}
