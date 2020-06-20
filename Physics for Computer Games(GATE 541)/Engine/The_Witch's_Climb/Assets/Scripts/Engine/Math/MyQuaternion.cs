using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace cyclone
{
    using real = System.Double;

    /**
     * Holds a three degree of freedom orientation.
     *
     * Quaternions have
     * several mathematical properties that make them useful for
     * representing orientations, but require four items of data to
     * hold the three degrees of freedom. These four items of data can
     * be viewed as the coefficients of a complex number with three
     * imaginary parts. The mathematics of the MyQuaternion is then
     * defined and is roughly correspondent to the math of 3D
     * rotations. A MyQuaternion is only a valid rotation if it is
     * normalised: i.e. it has a length of 1.
     *
     * @note Angular velocity and acceleration can be correctly
     * represented as vectors. Quaternions are only needed for
     * orientation.
     */
    class MyQuaternion
    {

        /**
        * Holds the real component of the MyQuaternion.
        */
        public real r;

        /**
         * Holds the first complex component of the
         * MyQuaternion.
         */
        public real i;

        /**
         * Holds the second complex component of the
         * MyQuaternion.
         */
        public real j;

        /**
         * Holds the third complex component of the
         * MyQuaternion.
         */
        public real k;



        /**
         * Holds the MyQuaternion data in array form.
         */
        //real data[4];


        // ... other MyQuaternion code as before ...

        /**
         * The default constructor creates a MyQuaternion representing
         * a zero rotation.
         */

        public MyQuaternion()
        {
        }

        /**
         * The explicit constructor creates a MyQuaternion with the given
         * components.
         *
         * @param r The real component of the rigid body's orientation
         * MyQuaternion.
         *
         * @param i The first complex component of the rigid body's
         * orientation MyQuaternion.
         *
         * @param j The second complex component of the rigid body's
         * orientation MyQuaternion.
         *
         * @param k The third complex component of the rigid body's
         * orientation MyQuaternion.
         *
         * @note The given orientation does not need to be normalised,
         * and can be zero. This function will not alter the given
         * values, or normalise the MyQuaternion. To normalise the
         * MyQuaternion (and make a zero MyQuaternion a legal rotation),
         * use the normalise function.
         *
         * @see normalise
         */
        public MyQuaternion(double r, double i, double j, double k)
        {
            this.r = r;
            this.i = i;
            this.j = j;
            this.k = k;
        }


        /**
         * Normalises the MyQuaternion to unit length, making it a valid
         * orientation MyQuaternion.
         */
        public void normalise()
        {
            real d = r * r + i * i + j * j + k * k;

            // Check for zero length MyQuaternion, and use the no-rotation
            // MyQuaternion in that case.
            if (d < Mathf.Epsilon)
            {
                r = 1;
                return;
            }

            d = ((real)1.0) / Mathf.Sqrt((float)d);
            r *= d;
            i *= d;
            j *= d;
            k *= d;
        }


        /**
         * Multiplies the MyQuaternion by the given MyQuaternion.
         *
         * @param multiplier The MyQuaternion by which to multiply.
         */
        public static MyQuaternion operator *(MyQuaternion lq, MyQuaternion multiplier)
        {

            lq.r = lq.r * multiplier.r - lq.i * multiplier.i -
                lq.j * multiplier.j - lq.k * multiplier.k;
            lq.i = lq.r * multiplier.i + lq.i * multiplier.r +
                lq.j * multiplier.k - lq.k * multiplier.j;
            lq.j = lq.r * multiplier.j + lq.j * multiplier.r +
                lq.k * multiplier.i - lq.i * multiplier.k;
            lq.k = lq.r * multiplier.k + lq.k * multiplier.r +
                lq.i * multiplier.j - lq.j * multiplier.i;

            return lq;
        }

        /**
         * Adds the given vector to this, scaled by the given amount.
         * This is used to update the orientation MyQuaternion by a rotation
         * and time.
         *
         * @param vector The vector to add.
         *
         * @param scale The amount of the vector to add.
         */
        public void addScaledVector(MyVector3 vector, real scale)
        {
            MyQuaternion q = new MyQuaternion(0,
                        vector.x * scale,
                        vector.y * scale,
                        vector.z * scale);
            q *= this;
            r += q.r * ((real)0.5);
            i += q.i * ((real)0.5);
            j += q.j * ((real)0.5);
            k += q.k * ((real)0.5);
        }

        public void rotateByVector(MyVector3 vector)
        {
            MyQuaternion q = new MyQuaternion(0, vector.x, vector.y, vector.z);
            MyQuaternion thisq = new MyQuaternion(this.r, this.i, this.j, this.k);
            thisq *= q;
            this.r = thisq.r;
            this.i = thisq.i;
            this.j = thisq.j;
            this.k = thisq.k;
        }
    };
}
