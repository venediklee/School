using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

namespace cyclone
{
    using real = System.Double;
    /**
     * The world represents an independent simulation of physics.  It
     * keeps track of a set of rigid bodies, and provides the means to
     * update them all.
     */
    class World : MonoBehaviour
    {
        // ... other World data as before ...
        /**
         * True if the world should calculate the number of iterations
         * to give the contact resolver at each frame.
         */
        bool calculateIterations;

        /**
         * Holds a single rigid body in a linked list of bodies.
         */
        class BodyRegistration
        {
            public RigidBody body;
            public BodyRegistration next;
        };

        /**
         * Holds the head of the list of registered bodies.
         */
        BodyRegistration firstBody;

        /**
         * Holds the resolver for sets of contacts.
         */
        ContactResolver resolver;

        /**
         * Holds one contact generators in a linked list.
         */
        class ContactGenRegistration
        {
            public ContactGenerator gen;
            public ContactGenRegistration next;
        };

        /**
         * Holds the head of the list of contact generators.
         */
        ContactGenRegistration firstContactGen;

        /**
         * Holds an array of contacts, for filling by the contact
         * generators.
         */
        Contact[] contacts;

        /**
         * Holds the maximum number of contacts allowed (i.e. the size
         * of the contacts array).
         */
        uint maxContacts;


        /**
         * Creates a new simulator that can handle up to the given
         * number of contacts per frame. You can also optionally give
         * a number of contact-resolution iterations to use. If you
         * don't give a number of iterations, then four times the
         * number of detected contacts will be used for each frame.
         */
        public World(uint maxContacts, uint iterations = 0)
        {
            contacts = new Contact[maxContacts];
            calculateIterations = (iterations == 0);
            //firstBody = new BodyRegistration();
            //firstContactGen = new ContactGenRegistration();
        }

        ~World()
        {
            contacts = null;
        }

        /**
         * Calls each of the registered contact generators to report
         * their contacts. Returns the number of generated contacts.
         */
        uint generateContacts()
        {
            uint limit = maxContacts;
            Contact nextContact = contacts[0];
            uint currentIndex = 0;
            ContactGenRegistration reg = firstContactGen;
            while (reg != null)
            {
                uint used = reg.gen.addContact(nextContact, limit);
                limit -= used;
                currentIndex += used;
                nextContact = contacts[currentIndex];

                // We've run out of contacts to fill. This means we're missing
                // contacts.
                if (limit <= 0) break;

                reg = reg.next;
            }

            // Return the number of contacts used.
            return maxContacts - limit;
        }

        /**
         * Processes all the physics for the world.
         */
        void runPhysics(real duration)
        {
            // First apply the force generators
            //registry.updateForces(duration);

            // Then integrate the objects
            BodyRegistration reg = firstBody;
            while (reg!=null)
            {
                // Remove all forces from the accumulator
                reg.body.integrate(duration);

                // Get the next registration
                reg = reg.next;
            }

            // Generate contacts
            uint usedContacts = generateContacts();

            // And process them
            if (calculateIterations) resolver.setIterations(usedContacts * 4);
            resolver.resolveContacts(contacts, usedContacts, duration);
        }


        /**
         * Initialises the world for a simulation frame. This clears
         * the force and torque accumulators for bodies in the
         * world. After calling this, the bodies can have their forces
         * and torques for this frame added.
         */
        void startFrame()
        {
            BodyRegistration reg = firstBody;
            while (reg!=null)
            {
                // Remove all forces from the accumulator
                reg.body.clearAccumulators();
                reg.body.calculateDerivedData();

                // Get the next registration
                reg = reg.next;
            }
        }

        private void Update()
        {
            startFrame();
        }

    };
}
