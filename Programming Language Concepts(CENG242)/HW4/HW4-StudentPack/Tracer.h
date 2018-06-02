#ifndef HW4_TRACER_H
#define HW4_TRACER_H


#include "Player.h"

class Tracer : public Player {
public:
    Tracer(uint id, int x, int y);

    // Name     : Tracer
    // Priority : { UP, LEFT, DOWN, RIGHT, ATTACK }

    // Armor    : BRICK
    // Weapon   : SHOVEL
    // HP       : 100

    // DO NOT MODIFY THE UPPER PART
    // ADD OWN PUBLIC METHODS/PROPERTIES/OVERRIDES BELOW
    ~Tracer();
    Armor getArmor() const;
    Weapon getWeapon() const ;
    /**
     * Every player has a different priority move list.
     * It's explained in the Players' header.
     *
     * @return The move priority list for the player.
     */
    std::vector<Move> getPriorityList() const ;
    /**
     * Get the full name of the player.
     *
     * Example (Tracer with ID 92) = "Tracer92"
     * Example (Tracer with ID 1)  = "Tracer01"
     *
     * @return Full name of the player.
     */
    const std::string getFullName() const;
    //bool operator()(Tracer a,Tracer b);
    
};


#endif //HW4_TRACER_H
