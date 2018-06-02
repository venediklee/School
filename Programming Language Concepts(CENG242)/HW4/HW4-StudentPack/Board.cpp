#include "Board.h"
#include "GameEngine.h"

/*
YOU MUST WRITE THE IMPLEMENTATIONS OF THE REQUESTED FUNCTIONS
IN THIS FILE. START YOUR IMPLEMENTATIONS BELOW THIS LINE 
*/
#include "Coordinate.h"
#include <algorithm>
Board::Board(uint newBoardSize, std::vector<Player *> *newPlayers): boardSize(newBoardSize),
                                                                    players(newPlayers),
                                                                    myStormWidth(0),
                                                                    myStormDamage(0)
{
    //std::sort(std::begin(*newPlayers),std::end(*newPlayers));
}

//TODO ERROR in here?
Board::~Board()=default;
//{
    //(*(this->players)).std::~vector<>();
//}

uint Board::getSize() const
{
    return this->boardSize;
}

bool Board::isCoordInBoard(const Coordinate& coord) const
{
    return coord.x < boardSize && coord.y < boardSize;
}

bool Board::isStormInCoord(const Coordinate &coord) const
{
    if(!isCoordInBoard(coord)) return false;
    else
    {
        //TODO ERROR? static?
        //int currStormWidth=Entity::stormWidthForRound(GameEngine::getCurrentRound());


        Coordinate minimal(myStormWidth,myStormWidth);
        Coordinate maximal(this->getSize()-myStormWidth-1,this->getSize()-myStormWidth-1);
        return !(minimal.x<=coord.x && coord.x<=maximal.x &&
                 minimal.y<=coord.y && coord.y<=maximal.y);
    }
}


bool Board::isCoordHill(const Coordinate& coord) const
{
    //if(this->getSize()%2==0) return false;
    return coord.x==(this->getSize())/2 && coord.y==(this->getSize())/2;
}


Player* Board::operator[](const Coordinate& coord) const
{
    if(!this->isCoordInBoard(coord)) return nullptr;
    else
    {
        int i=0;//used for iteration
        int vectorSize=this->players->size();
        while(i<vectorSize)
        {
            if(this->players->at(i)->getCoord()==coord) return this->players->at(i);
            i++;
        }
        return nullptr;
    }

}

Coordinate Board::calculateCoordWithMove(Move move, const Coordinate &coord) const
{
    Coordinate newCoord(coord.x,coord.y);
    switch(move)
    {
        case UP: newCoord.y-=1;
            break;
        case DOWN: newCoord.y+=1;
            break;
        case LEFT: newCoord.x-=1;
            break;
        case RIGHT: newCoord.x+=1;
            break;
        default: return coord;//ATTACK and NOOP
    }
    if(!this->isCoordInBoard(newCoord) || (*this)[newCoord]!= nullptr) return coord;
    else
    {
        //TODO ERROR(?) should I move the player in the coord or not?
        return newCoord;
    }

}

std::vector<Coordinate> Board::visibleCoordsFromCoord(const Coordinate &coord) const
{
    std::vector<Coordinate> visibleCoords;
    if(!this->isCoordInBoard(coord)) return visibleCoords;//If the given coordinate is not in board
    else
    {
        if(this->isCoordInBoard(coord+UP)) visibleCoords.push_back(coord+UP);
        if(this->isCoordInBoard(coord+DOWN)) visibleCoords.push_back(coord+DOWN);
        if(this->isCoordInBoard(coord+RIGHT)) visibleCoords.push_back(coord+RIGHT);
        if(this->isCoordInBoard(coord+LEFT)) visibleCoords.push_back(coord+LEFT);
    }

}


//TODO update myStormWidth
/**
 * Calculate the storm according to the currentRound.
 *
 * @param currentRound The current round being played.
 */
void Board::updateStorm(uint currentRound)
{
    myStormWidth=(Entity::stormWidthForRound(currentRound)>this->getSize()/2)?this->getSize()/2
            :Entity::stormWidthForRound(currentRound);
    myStormDamage=Entity::stormDamageForRound(currentRound);
}

std::vector<Player *> Board::getPlayers() const
{
    return *this->players;
}


void Board::deletePlayer(int locationAtVector)
{
    this->players->erase(std::begin(*(this->players))+locationAtVector);
}