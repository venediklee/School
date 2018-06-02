#include "GameEngine.h"

/*
YOU MUST WRITE THE IMPLEMENTATIONS OF THE REQUESTED FUNCTIONS
IN THIS FILE. START YOUR IMPLEMENTATIONS BELOW THIS LINE 
*/
#include <iostream>
#include <algorithm>


//TODO delete players from the board location and players vector(s?)

GameEngine::GameEngine(uint boardSize, std::vector<Player *> *players):board(Board(boardSize,players)),
                                                                       currentRound(0),stormWidth(0){};
//TODO MEMO LEAK?
GameEngine::~GameEngine()=default;
/*{
    delete board;//maybe delete board->getPlayers or someshit like that?
}*/

const Board& GameEngine::getBoard() const
{
    return this->board;
}


Player* GameEngine::operator[](uint id) const
{
    std::vector<Player *> allPlayers=this->board.getPlayers();
    int i=0;//used for iteration
    int vectorSize=allPlayers.size();
    while(i<vectorSize)
    {
        if(allPlayers.at(i)->getID()==id) return allPlayers.at(i);
        i++;
    }
    return nullptr;
}

//TODO utilize getWinnerPlayer?
bool GameEngine::isFinished() const
{
    //TODO don't forget that you should delete players when they die xd
    return this->getBoard().getPlayers().size() == 0 ||
           (this->getBoard().getPlayers().size()==1 &&
            this->getBoard().isCoordHill(this->board.getPlayers().at(0)->getCoord()));
}

void GameEngine::takeTurn()
{
    ++currentRound;
    std::cout<<"-- START ROUND "<<currentRound<<" --"<<std::endl;
    //TODO utilize takeTurnForPlayer
    //TODO utilize getWinnerPlayer

    this->board.updateStorm(currentRound);


    for(int i=0;i<this->board.getPlayers().size();++i)
    {

        this->takeTurnForPlayer(this->board.getPlayers().at(i)->getID());
        if(didPreviousPlayerDie) --i;
        didPreviousPlayerDie=0;
        
    }
    std::cout<<"-- END ROUND "<<currentRound<<" --"<<std::endl;
}


Move GameEngine::takeTurnForPlayer(uint playerID)
{

    Player* currPlayer=(*this)[playerID];
    std::vector<Move> priorityList=currPlayer->getPriorityList();
    std::vector<Coordinate> visCoords=this->board.visibleCoordsFromCoord(currPlayer->getCoord());

    int roundDamage=Entity::stormDamageForRound(currentRound);

    if(this->board.isStormInCoord(currPlayer->getCoord()))//If the player is in the storm
    {
        std::cout<<currPlayer->getFullName()
                 <<"("<<currPlayer->getHP()<<")"<<
                 " is STORMED! (-"
                 <<roundDamage<<")"<<
                 std::endl;
        //damage player
        currPlayer->setHP(currPlayer->getHP()-roundDamage);
    }

    if(currPlayer->getHP()<=0)//if dead by storm
    {
        std::cout<<currPlayer->getFullName()
                 <<"("<<currPlayer->getHP()<<") DIED."<<
                 std::endl;
        int i = 0;
        for (i=0; i < this->getBoard().getPlayers().size(); ++i) if(currPlayer==this->getBoard().getPlayers().at(i)) break;
        this->board.deletePlayer(i);
        didPreviousPlayerDie=1;

        return NOOP;
    }
    for (int i = 0; i < priorityList.size(); ++i)
    {
        Move myMove=priorityList[i];
        Coordinate currPlayerCoord=currPlayer->getCoord();
        if(myMove==NOOP) return NOOP;
        else if(myMove==ATTACK)
        {
            std::vector<Player*> attPlayers;
            Player* upPlayer=this->board[currPlayerCoord+UP];
            Player* downPlayer=this->board[currPlayerCoord+DOWN];
            Player* rightPlayer=this->board[currPlayerCoord+RIGHT];
            Player* leftPlayer=this->board[currPlayerCoord+LEFT];

            if(upPlayer!= nullptr) attPlayers.push_back(upPlayer);
            if(downPlayer!= nullptr) attPlayers.push_back(downPlayer);
            if(rightPlayer!= nullptr) attPlayers.push_back(rightPlayer);
            if(leftPlayer!= nullptr) attPlayers.push_back(leftPlayer);

            std::sort(std::begin(attPlayers),std::end(attPlayers));
            if(attPlayers.empty()) continue;//no need to check the rest if there is no one to attack
            //std::sort(std::begin(attPlayers),std::end(attPlayers));//now first element is the one to attack
            currPlayer->attackTo(attPlayers.at(0));
            if(attPlayers.at(0)->isDead())//if other player died
            {
                //announce death
                std::cout<<attPlayers.at(0)->getFullName()
                         <<"("<<attPlayers.at(0)->getHP()<<") DIED."<<
                         std::endl;
                //remove the dead player from board


                for (int j = 0; j <this->getBoard().getPlayers().size() ; ++j)
                {
                    if(attPlayers.at(0)==this->getBoard().getPlayers().at(j))
                    {
                        if(currPlayer->getID()>attPlayers.at(0)->getID()) this->didPreviousPlayerDie=1;
                        this->board.deletePlayer(j);
                        return ATTACK;
                    }
                }
                //this->board.getPlayers().erase(std::begin(this->board.getPlayers())+1);
                //std::cout<<"TEST"<<std::endl;
            }
            return ATTACK;

        }
        else//(UP/DOWN/LEFT/RIGHT):
        {

            Coordinate possibleNewCoord=this->board.
                    calculateCoordWithMove(priorityList[i],currPlayerCoord);

            if(possibleNewCoord==currPlayerCoord) continue;//if the player can't move skip
            else
            {
                //if player getting away from the hill skip
                Coordinate myHillCoord=Coordinate (this->getBoard().getSize()/2,this->getBoard().getSize()/2);
                int possibleDistance=abs(myHillCoord.x-possibleNewCoord.x)+abs(myHillCoord.y-possibleNewCoord.y);
                int normalDistance=abs(myHillCoord.x-currPlayerCoord.x)+abs(myHillCoord.y-currPlayerCoord.y);
                if(normalDistance<possibleDistance) continue;

                currPlayer->executeMove(priorityList[i]);
                return priorityList[i];
            }
        }

    }

    return NOOP;// If the priority list is exhausted;
}

/**
 * Find the winner player.
 *
 * nullptr if there are 0 players left, or the game isn't finished yet.
 *
 * @return The winner player.
 */
Player* GameEngine::getWinnerPlayer() const
{
    if(this->board.getPlayers().size()==1)
    {
        if(this->board.isCoordHill(this->board.getPlayers().at(0)->getCoord()))
            if(this->board.getPlayers().at(0)->getHP()>0)
                return this->board.getPlayers().at(0);
    }
    return nullptr;
}

// DO NOT MODIFY THE UPPER PART
// ADD OWN PUBLIC METHODS/PROPERTIES BELOW
