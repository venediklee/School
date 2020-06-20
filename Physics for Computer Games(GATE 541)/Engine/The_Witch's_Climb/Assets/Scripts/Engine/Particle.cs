using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace cyclone
{
    
    public class Particle //: MonoBehaviour
    {
        //default constructor
        public Particle()
        {
            position = new MyVector3();
            velocity = new MyVector3();
            forceAccum = new MyVector3();
            acceleration = new MyVector3();
        }

        /**
         * Holds the inverse of the mass of the particle. It
         * is more useful to hold the inverse mass because
         * integration is simpler, and because in real time
         * simulation it is more useful to have objects with
         * infinite mass (immovable) than zero mass
         * (completely unstable in numerical simulation).
         */
        public double inverseMass { get; set; }

        /**
         * Holds the amount of damping applied to linear
         * motion. Damping is required to remove energy added
         * through numerical instability in the integrator.
         */
        public double damping { get; set; }

        /**
         * Holds the linear position of the particle in
         * world space.
         */
        public MyVector3 position { get; set; }

        /**
         * Holds the linear velocity of the particle in
         * world space.
         */
        public MyVector3 velocity { get; set; }

        /**
         * Holds the accumulated force to be applied at the next
         * simulation iteration only. This value is zeroed at each
         * integration step.
         */
        public MyVector3 forceAccum { get; set; }

        /**
         * Holds the acceleration of the particle.  This value
         * can be used to set acceleration due to gravity (its primary
         * use), or any other constant acceleration.
         */
        public MyVector3 acceleration { get; set; }


        /**
         * Integrates the particle forward in time by the given amount.
         * This function uses a Newton-Euler integration method, which is a
         * linear approximation to the correct integral. For this reason it
         * may be inaccurate in some cases.
         */
        public void Integrate(double duration)
        {
            // We don't integrate things with zero mass.
            if (inverseMass <= 0.0f) return;

            Debug.Assert(duration > 0.0);

            // Update linear position.
            position.AddScaledVector(this.velocity, duration);

            // Work out the acceleration from the force
            MyVector3 resultingAcc = acceleration;
            resultingAcc.AddScaledVector(forceAccum, inverseMass);

            // Update linear velocity from the acceleration.
            velocity.AddScaledVector(resultingAcc, duration);

            // Impose drag.
            velocity *= System.Math.Pow(damping, duration);

            // Clear the forces.
            ClearAccumulator();
        }

        /**
         * Sets the mass of the particle.
         *
         * @param mass The new mass of the body. This may not be zero.
         * Small masses can produce unstable rigid bodies under
         * simulation.
         *
         * @warning This invalidates internal data for the particle.
         * Either an integration function, or the calculateInternals
         * function should be called before trying to get any settings
         * from the particle.
         */
        public void SetMass(double mass)
        {
            Debug.Assert(mass != 0);
                inverseMass = ((double)1.0)/mass;
            //Debug.Log("setting" + mass + "mass for particle");
        }

        /**
         * Gets the mass of the particle.
         *
         * @return The current mass of the particle.
         */
        public double GetMass()
        {
            if (inverseMass == 0) 
            {
                return double.MaxValue;
            } 
            else 
            {
                return ((double)1.0)/inverseMass;
            }
        }


        /**
         * Sets the inverse mass of the particle.
         *
         * @param inverseMass The new inverse mass of the body. This
         * may be zero, for a body with infinite mass
         * (i.e. unmovable).
         *
         * @warning This invalidates internal data for the particle.
         * Either an integration function, or the calculateInternals
         * function should be called before trying to get any settings
         * from the particle.
         */
        public void SetInverseMass(double inverseMass)
        {
            this.inverseMass = inverseMass;
        }

        /**
         * Gets the inverse mass of the particle.
         *
         * @return The current inverse mass of the particle.
         */
        public double GetInverseMass()
        {
            return inverseMass;
        }

        /**
         * Returns true if the mass of the particle is not-infinite.
         */
        public bool HasFiniteMass()
        {
            return inverseMass >= 0.0f;
        }

        /**
         * Sets both the damping of the particle.
         */
        public void SetDamping(double damping)
        {
            this.damping = damping;
        }

        /**
         * Gets the current damping value.
         */
        public double GetDamping()
        {
            double d = new double();
            d = damping;
            return d;
        }

        /**
         * Sets the position of the particle.
         *
         * @param position The new position of the particle.
         */
        public void SetPosition(MyVector3 position)
        {
            this.position = position;
        }

        /**
         * Sets the position of the particle by component.
         *
         * @param x The x coordinate of the new position of the rigid
         * body.
         *
         * @param y The y coordinate of the new position of the rigid
         * body.
         *
         * @param z The z coordinate of the new position of the rigid
         * body.
         */
        public void SetPosition(double x,double y, double z)
        {
            position.x = x;
            position.y = y;
            position.z = z;
        }

        /**
         * Fills the given vector with the position of the particle.
         *
         * @param position A pointer to a vector into which to write
         * the position.
         */
        public void GetPosition(MyVector3 position)
        {
            position = this.position;
        }

        /**
         * Gets the position of the particle.
         *
         * @return The position of the particle.
         */
        public MyVector3 GetPosition()
        {
            return new MyVector3(position);
        }

        /**
         * Sets the velocity of the particle.
         *
         * @param velocity The new velocity of the particle.
         */
        public void SetVelocity(MyVector3 velocity)
        {
            this.velocity = velocity;
        }

        /**
         * Sets the velocity of the particle by component.
         *
         * @param x The x coordinate of the new velocity of the rigid
         * body.
         *
         * @param y The y coordinate of the new velocity of the rigid
         * body.
         *
         * @param z The z coordinate of the new velocity of the rigid
         * body.
         */
        public void SetVelocity(double x, double y, double z)
        {
            velocity.x = x;
            velocity.y = y;
            velocity.z = z;
        }

        /**
         * Fills the given vector with the velocity of the particle.
         *
         * @param velocity A pointer to a vector into which to write
         * the velocity. The velocity is given in world local space.
         */
        public void GetVelocity(MyVector3 velocity)
        {
            velocity = this.velocity;
        }

        /**
         * Gets the velocity of the particle.
         *
         * @return The velocity of the particle. The velocity is
         * given in world local space.
         */
        public MyVector3 GetVelocity()
        {
            return new MyVector3(velocity);
        }

        /**
        * Sets the constant acceleration of the particle.
        *
        * @param acceleration The new acceleration of the particle.
        */
        public void SetAcceleration(MyVector3 acceleration)
        {
            this.acceleration = acceleration;
        }

        /**
         * Sets the constant acceleration of the particle by component.
         *
         * @param x The x coordinate of the new acceleration of the rigid
         * body.
         *
         * @param y The y coordinate of the new acceleration of the rigid
         * body.
         *
         * @param z The z coordinate of the new acceleration of the rigid
         * body.
         */
        public void SetAcceleration(double x,double y, double z)
        {
            acceleration.x = x;
            acceleration.y = y;
            acceleration.z = z;
        }

        /**
         * Fills the given vector with the acceleration of the particle.
         *
         * @param acceleration A pointer to a vector into which to write
         * the acceleration. The acceleration is given in world local space.
         */
        public void GetAcceleration(MyVector3 acceleration)
        {
            acceleration = this.acceleration;
        }


        /**
         * Gets the acceleration of the particle.
         *
         * @return The acceleration of the particle. The acceleration is
         * given in world local space.
         */
        public MyVector3 GetAcceleration()
        {
            return new MyVector3(acceleration);
        }

        /**
         * Clears the forces applied to the particle. This will be
         * called automatically after each integration step.
         */
        public void ClearAccumulator()
        {
            forceAccum.Clear();
        }

        /**
         * Adds the given force to the particle, to be applied at the
         * next iteration only.
         *
         * @param force The force to apply.
         */
        public void AddForce(MyVector3 force)
        {
            forceAccum += force;
        }

        public override bool Equals(object obj)
        {
            var particle = obj as Particle;
            return particle != null &&
                   inverseMass == particle.inverseMass &&
                   damping == particle.damping &&
                   EqualityComparer<MyVector3>.Default.Equals(position, particle.position) &&
                   EqualityComparer<MyVector3>.Default.Equals(velocity, particle.velocity) &&
                   EqualityComparer<MyVector3>.Default.Equals(forceAccum, particle.forceAccum) &&
                   EqualityComparer<MyVector3>.Default.Equals(acceleration, particle.acceleration);
        }

        public override int GetHashCode()
        {
            var hashCode = -1296583852;
            hashCode = hashCode * -1521134295 + inverseMass.GetHashCode();
            hashCode = hashCode * -1521134295 + damping.GetHashCode();
            hashCode = hashCode * -1521134295 + EqualityComparer<MyVector3>.Default.GetHashCode(position);
            hashCode = hashCode * -1521134295 + EqualityComparer<MyVector3>.Default.GetHashCode(velocity);
            hashCode = hashCode * -1521134295 + EqualityComparer<MyVector3>.Default.GetHashCode(forceAccum);
            hashCode = hashCode * -1521134295 + EqualityComparer<MyVector3>.Default.GetHashCode(acceleration);
            return hashCode;
        }

        //FINISHED original particle class


        public void SetPosition(Vector3 v)
        {
            position.x = v.x;
            position.y = v.y;
            position.z = v.z;
        }

        /// <summary>
        /// decreases velocity after given time
        /// </summary>
        /// <param name="time"></param>
        /// <param name="percent">how much percent of the velocity will be decreased</param>
        /// <returns></returns>
        public IEnumerator DecreaseVelocityAfterTime(float time, float percent = 100)
        {
            yield return new WaitForSecondsRealtime(time);
            this.SetVelocity(this.GetVelocity() * (100 - percent));
            this.acceleration *= 0;
        }
        /// <summary>
        /// decreases acceleration after given time
        /// </summary>
        /// <param name="time"></param>
        /// <param name="percent">how much percent of the velocity will be decreased</param>
        /// <returns></returns>
        public IEnumerator DecreaseAccelerationAfterTime(float time, float percent = 100)
        {
            yield return new WaitForSecondsRealtime(time);
            this.SetAcceleration(this.GetAcceleration() * (100 - percent));
        }
    };
}
