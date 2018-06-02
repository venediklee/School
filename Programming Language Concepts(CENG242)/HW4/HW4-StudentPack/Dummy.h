#ifndef HW4_DUMMY_H
#define HW4_DUMMY_H


#include "Player.h"

class Dummy : public Player {
public:
    Dummy(uint id, int x, int y);

    // Name     : Dummy
    // Priority : { NOOP }

    // Armor    : NOARMOR
    // Weapon   : NOWEAPON
    // HP       : 1000

    // DO NOT MODIFY THE UPPER PART
    // ADD OWN PUBLIC METHODS/PROPERTIES/OVERRIDES BELOW


    ~Dummy();
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
    //bool operator()(Dummy a,Dummy b);

};


#endif //HW4_DUMMY_H
