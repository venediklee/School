#include "GameParser.h"

/*
YOU MUST WRITE THE IMPLEMENTATIONS OF THE REQUESTED FUNCTIONS
IN THIS FILE. START YOUR IMPLEMENTATIONS BELOW THIS LINE
*/
#include <fstream>
#include <iostream>
#include <cstring>
#include <sstream>
#include <algorithm>
#include "Ambusher.h"
#include "Berserk.h"
#include "Dummy.h"
#include "Pacifist.h"
#include "Tracer.h"

/**
     * Parse the file with given name and create players accordingly.
     *
     * GameParser DOES NOT have any responsibility over these Players.
     *
     * Note: The file will always exists, and there will be no erroneous input.
     *
     * @param filename The name of the file to be parsed.
     * @return  pair.first: Board size.
     *          pair.second: The vector of the constructed players.
     */
std::pair<int, std::vector<Player *> *> GameParser::parseFileWithName(const std::string& filename)
{
    std::vector<Player*> players;
    int BoardSize;
    std::ifstream myFile(filename.c_str());
    std::string line;
    std::stringstream dummyStream;

    if(!myFile.is_open())
    {
        std::cerr<<"you done fucked up"<<std::endl;


    }
    else
    {
        while(std::getline(myFile,line))//read stuff from the file
        {
            if(line.find("Board"))//first input line
            {

                std::string BoardSizeStr=line.substr(12,std::string::npos);
                dummyStream.str(std::string());
                dummyStream<<BoardSizeStr;
                dummyStream>>BoardSize;
                //BoardSize=std::stoi(BoardSizeStr);
            }
            else if (line.find("Player"))//2'nd line where total number of players resides.
            {
                //I don't use this right now.
            }
            else//lines where players' information is given
            {
                uint newID=0;
                int x=0,y=0;
                std::string firstParse=line.substr(0,line.find("::"));//this is the id
                std::string nextOfFirstParse=line.substr(line.find("::")+2);//this is -Name-::x::y

                std::string secondParse=nextOfFirstParse.substr(nextOfFirstParse.find("::")+2,
                                                              nextOfFirstParse.find_last_of("::"));
                                                                //this is x
                std::string lastParse=nextOfFirstParse.substr(nextOfFirstParse.find_last_of("::"));
                                                                //this is y


                //get the ID



                dummyStream.str(std::string());
                dummyStream<<firstParse;
                dummyStream>>newID;
                //newID=std::stoi(firstParse);

                //find x and y coordinates
                dummyStream.str(std::string());
                dummyStream<<secondParse;
                dummyStream>>x;
                //x=std::stoi(secondParse);
                dummyStream.str(std::string());
                dummyStream<<lastParse;
                dummyStream>>y;
                //y=std::stoi(lastParse);
                //lines of players
                if(line.find("Ambusher"))
                {
                    players.push_back(new Ambusher(newID,x,y));
                }
                else if(line.find("Berserk"))
                {
                    players.push_back(new Berserk(newID,x,y));
                }
                else if(line.find("Dummy"))
                {
                    players.push_back(new Dummy(newID,x,y));
                }
                else if(line.find("Pacifist"))
                {
                    players.push_back(new Pacifist(newID,x,y));
                }
                else if(line.find("Tracer"))
                {
                    players.push_back(new Tracer(newID,x,y));
                }
                else
                {
                    std::cout<<std::endl<<"SELF ERROR:::LINES EXCEEDED"<<std::endl;
                }
            }
        }
        myFile.close();

    }
    std::sort(std::begin(players),std::end(players));
    return std::make_pair(BoardSize,&players);


};
