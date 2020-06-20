using System.Collections;
using System.Collections.Generic;
using UnityEngine;



namespace cyclone
{
    using Registry = System.Collections.Generic.List<cyclone.ParticleForceRegistry.ParticleForceRegistration>;

    /**
     * A force generator can be asked to add a force to one or more
     * particles.
     */
    class ParticleForceGenerator
    {
        //TODO check for errors
        /**
         * Overload this in implementations of the interface to calculate
         * and update the force applied to the given particle.
         */
        public virtual void updateForce(Particle particle, double duration) { }
    };


    /**
     * A force generator that applies a gravitational force. One instance
     * can be used for multiple particles.
     */
    class ParticleGravity : ParticleForceGenerator
    {
        /** Holds the acceleration due to gravity. */
        MyVector3 gravity;



        /** Creates the generator with the given acceleration. */
        public ParticleGravity(MyVector3 gravity)
        {
            this.gravity = gravity;
        }

        /** Applies the gravitational force to the given particle. */
        public virtual void updateForce(Particle particle, double duration)
        {
            // Check that we do not have infinite mass
            if (!particle.HasFiniteMass()) return;

            // Apply the mass-scaled force to the particle
            particle.AddForce(gravity * particle.GetMass());
        }
    };

    /**
     * A force generator that applies a drag force. One instance
     * can be used for multiple particles.
     */
    class ParticleDrag : ParticleForceGenerator
    {
        /** Holds the velocity drag coeffificent. */
        double k1;

        /** Holds the velocity squared drag coeffificent. */
        double k2;


        /** Creates the generator with the given coefficients. */
        public ParticleDrag(double k1, double k2)
        {
            this.k1 = k1;
            this.k2 = k2;
        }

        /** Applies the drag force to the given particle. */
        public virtual void updateForce(Particle particle, double duration)
        {
            MyVector3 force = particle.GetVelocity();

            // Calculate the total drag coefficient
            double dragCoeff = force.Magnitude();
            dragCoeff = k1 * dragCoeff + k2 * dragCoeff * dragCoeff;

            // Calculate the final force and apply it
            force.Normalise();
            force *= -dragCoeff;
            particle.AddForce(force);
        }
    };


    /**
     * A force generator that applies a Spring force, where
     * one end is attached to a fixed point in space.
     */
    class ParticleAnchoredSpring :  ParticleForceGenerator
    {

        /** The location of the anchored end of the spring. */
        protected MyVector3 anchor;

        /** Holds the sprint constant. */
        protected double springConstant;

        /** Holds the rest length of the spring. */
        protected double restLength;

        public ParticleAnchoredSpring()
        {
        }

        /** Creates a new spring with the given parameters. */
        public ParticleAnchoredSpring(MyVector3 anchor, double springConstant, double restLength)
        {
            this.anchor = anchor;
            this.springConstant = springConstant;
            this.restLength = restLength;
        }

        /** Retrieve the anchor point. */
        public MyVector3 getAnchor()
        {
            return new MyVector3(anchor);
        }

        /** Set the spring's properties. */
        public void init(MyVector3 anchor, double springConstant, double restLength)
        {
            this.anchor = anchor;
            this.springConstant = springConstant;
            this.restLength = restLength;
        }

        /** Applies the spring force to the given particle. */
        public virtual void updateForce(Particle particle, double duration)
        {
            // Calculate the vector of the spring
            MyVector3 force = particle.GetPosition();
            force -= anchor;

            // Calculate the magnitude of the force
            double magnitude = force.Magnitude();
            magnitude = (restLength - magnitude) * springConstant;

            // Calculate the final force and apply it
            force.Normalise();
            force *= magnitude;
            particle.AddForce(force);
        }
    };

    /**
    * A force generator that applies a bungee force, where
    * one end is attached to a fixed point in space.
    */
    class ParticleAnchoredBungee : ParticleAnchoredSpring
    {
        public ParticleAnchoredBungee()
        {
        }

        public ParticleAnchoredBungee(MyVector3 anchor, double springConstant, double restLength) : base(anchor, springConstant, restLength)
        {
        }

        /** Applies the spring force to the given particle. */
        public virtual void updateForce(Particle particle, double duration)
        {
            // Calculate the vector of the spring
            MyVector3 force = particle.GetPosition();
            force -= anchor;

            // Calculate the magnitude of the force
            double magnitude = force.Magnitude();
            if (magnitude < restLength) return;

            magnitude = magnitude - restLength;
            magnitude *= springConstant;

            // Calculate the final force and apply it
            force.Normalise();
            force *= -magnitude;
            particle.AddForce(force);
        }
    };


    /**
     * A force generator that fakes a stiff spring force, and where
     * one end is attached to a fixed point in space.
     */
    class ParticleFakeSpring : ParticleForceGenerator
    {
        /** The location of the anchored end of the spring. */
        MyVector3 anchor;

        /** Holds the sprint constant. */
        double springConstant;

        /** Holds the damping on the oscillation of the spring. */
        double damping;




        /** Creates a new spring with the given parameters. */
        public ParticleFakeSpring(MyVector3 anchor, double springConstant, double damping)
        {
            this.anchor = anchor;
            this.springConstant = springConstant;
            this.damping = damping;
        }
        
        /** Applies the spring force to the given particle. */
        public virtual void updateForce(Particle particle, double duration)
        {
            // Check that we do not have infinite mass
            if (!particle.HasFiniteMass()) return;

            // Calculate the relative position of the particle to the anchor
            MyVector3 position = particle.GetPosition();
            position -= anchor;

            //TODO precision: double to float in all mathf functions
            // Calculate the constants and check they are in bounds.
            double gamma = 0.5f * Mathf.Sqrt((float)(4 * springConstant - damping * damping));
            if (gamma == 0.0f) return;
            MyVector3 c = position * (damping / (2.0f * gamma)) +
                particle.GetVelocity() * (1.0f / gamma);

            // Calculate the target position
            MyVector3 target = position * Mathf.Cos((float)(gamma * duration)) +
                c * Mathf.Sin((float)(gamma * duration));
            target *= Mathf.Exp((float)(-0.5f * duration * damping));

            // Calculate the resulting acceleration and therefore the force
            MyVector3 accel = (target - position) * ((double)1.0 / (duration * duration)) -
                particle.GetVelocity() * ((double)1.0 / duration);
            particle.AddForce(accel * particle.GetMass());
        }
    };

    /**
     * A force generator that applies a Spring force.
     */
    class ParticleSpring :  ParticleForceGenerator
    {
        /** The particle at the other end of the spring. */
        Particle other;

        /** Holds the sprint constant. */
        double springConstant;

        /** Holds the rest length of the spring. */
        double restLength;
        
        /** Creates a new spring with the given parameters. */
        public ParticleSpring(Particle other, double springConstant, double restLength)
        {
            this.other = other;
            this.springConstant = springConstant;
            this.restLength = restLength;
        }

        /** Applies the spring force to the given particle. */
        public virtual void updateForce(Particle particle, double duration)
        {
            // Calculate the vector of the spring
            MyVector3 force = particle.GetPosition();
            force -= other.GetPosition();

            // Calculate the magnitude of the force
            double magnitude = force.Magnitude();
            magnitude = Mathf.Abs((float)(magnitude - restLength));
            magnitude *= springConstant;

            // Calculate the final force and apply it
            force.Normalise();
            force *= -magnitude;
            particle.AddForce(force);
        }
    };

    /**
     * A force generator that applies a spring force only
     * when extended.
     */
    class ParticleBungee :  ParticleForceGenerator
    {
        /** The particle at the other end of the spring. */
        Particle other;

        /** Holds the sprint constant. */
        double springConstant;

        /**
         * Holds the length of the bungee at the point it begins to
         * generator a force.
         */
        double restLength;

        /** Creates a new bungee with the given parameters. */
        public ParticleBungee(Particle other, double springConstant, double restLength)
        {
            this.other = other;
            this.springConstant = springConstant;
            this.restLength = restLength;
        }

        /** Applies the spring force to the given particle. */
        public virtual void updateForce(Particle particle, double duration)
        {
            // Calculate the vector of the spring
            MyVector3 force = particle.GetPosition();
            force -= other.GetPosition();

            // Check if the bungee is compressed
            double magnitude = force.Magnitude();
            if (magnitude <= restLength) return;

            // Calculate the magnitude of the force
            magnitude = springConstant * (restLength - magnitude);

            // Calculate the final force and apply it
            force.Normalise();
            force *= -magnitude;
            particle.AddForce(force);
        }
    };

    /**
     * A force generator that applies a buoyancy force for a plane of
     * liquid parrallel to XZ plane.
     */
    class ParticleBuoyancy :  ParticleForceGenerator
    {
        /**
         * The maximum submersion depth of the object before
         * it generates its maximum boyancy force.
         */
        double maxDepth;

        /**
         * The volume of the object.
         */
        double volume;

        /**
         * The height of the water plane above y=0. The plane will be
         * parrallel to the XZ plane.
         */
        double waterHeight;

        /**
         * The density of the liquid. Pure water has a density of
         * 1000kg per cubic meter.
         */
        double liquidDensity;

        /** Creates a new buoyancy force with the given parameters. */
        public ParticleBuoyancy(double maxDepth, double volume, double waterHeight, double liquidDensity = 1000)
        {
            this.maxDepth = maxDepth;
            this.volume = volume;
            this.waterHeight = waterHeight;
            this.liquidDensity = liquidDensity;
        }


        /** Applies the buoyancy force to the given particle. */
        public virtual void updateForce(Particle particle, double duration)
        {
            // Calculate the submersion depth
            double depth = particle.GetPosition().y;

            // Check if we're out of the water
            if (depth >= waterHeight + maxDepth) return;
            MyVector3 force= new MyVector3(0,0,0);

            // Check if we're at maximum depth
            if (depth <= waterHeight - maxDepth)
            {
                force.y = liquidDensity * volume;
                particle.AddForce(force);
                return;
            }

            // Otherwise we are partly submerged
            force.y = liquidDensity * volume *
                (depth - maxDepth - waterHeight) / 2 * maxDepth;
            particle.AddForce(force);
        }
    };

    /**
     * Holds all the force generators and the particles they apply to.
     */
    class ParticleForceRegistry
    {
        /**
         * Keeps track of one force generator and the particle it
         * applies to.
         */
        //TODO make this protected
        public struct ParticleForceRegistration
        {
            //TODO these are not public in cyclone 
            public Particle particle;
            public ParticleForceGenerator fg;

            public ParticleForceRegistration(Particle particle, ParticleForceGenerator fg)
            {
                this.particle = particle;
                this.fg = fg;
            }
        };

        /**
         * Holds the list of registrations.
         */
        protected Registry registrations;


        /**
         * Registers the given force generator to apply to the
         * given particle.
         */
        public void add(Particle particle, ParticleForceGenerator fg)
        {
            ParticleForceRegistration registration = new ParticleForceRegistration();
            registration.particle = particle;
            registration.fg = fg;
            registrations.Add(registration);
        }

        /**
         * Removes the given registered pair from the registry.
         * If the pair is not registered, this method will have
         * no effect.
         */
        public void remove(Particle particle, ParticleForceGenerator fg)
        {
            //TODO also missing in cyclone
        }

        /**
         * Clears all registrations from the registry. This will
         * not delete the particles or the force generators
         * themselves, just the records of their connection.
         */
        public void clear()
        {
            //TODO also missing in cyclone
        }

        /**
         * Calls all the force generators to update the forces of
         * their corresponding particles.
         */
        public void updateForces(double duration)
        {
            foreach (ParticleForceRegistration it in registrations)
            {
                it.fg.updateForce(it.particle, duration);
            }
        }
    };
}
