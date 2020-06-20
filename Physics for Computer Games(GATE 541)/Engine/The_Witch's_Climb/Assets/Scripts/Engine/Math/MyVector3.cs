using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace cyclone
{
    
    /**
    * Holds a vector in 3 dimensions. Four data members are allocated
    * to ensure alignment in an array.
    *
    * @note This class contains a lot of inline methods for basic
    * mathematics. The implementations are included in the header
    * file.
    */
    public class MyVector3 //: MonoBehaviour
    {
        /** Holds the value along the x axis. */
        public double x;

        /** Holds the value along the y axis. */
        public double y;

        /** Holds the value along the z axis. */
        public double z;

        /** Padding to ensure 4 word alignment. */
        readonly double pad;

        /** The default constructor creates a zero vector. */
        public MyVector3()
        {
            x = 0;
            y = 0;
            z = 0;
        }

        /**
         * The explicit constructor creates a vector with the given
         * components.
         */
        public MyVector3(double nx, double ny, double nz)
        {
            x = nx;
            y = ny;
            z = nz;
        }

        //THESE ARE DEFAULT VALUES OF MyVector3 class, can change them per object in IParticle.cs class
        //DEV_NOTE :: dropped const keyword on all the following variables
        //static MyVector3 GRAVITY = new MyVector3(0, -9.81f, 0);
        //static MyVector3 HIGH_GRAVITY = new MyVector3(0, -19.62f, 0);
        //static MyVector3 UP = new MyVector3(0, 1, 0);
        //static MyVector3 RIGHT = new MyVector3(1, 0, 0);
        //static MyVector3 OUT_OF_SCREEN = new MyVector3(0, 0, 1);
        //static MyVector3 X = new MyVector3(0, 1, 0);
        //static MyVector3 Y = new MyVector3(1, 0, 0);
        //static MyVector3 Z = new MyVector3(0, 0, 1);

        // ... Other MyVector3 code as before ...

        //DEV_NOTE :: this is operator[] (there is no double&)-- NOT TESTED
        public double this[int key]
        {
            get
            {
                if (key == 0) return x;
                if (key == 1) return y;
                return z;
            }
        }

        //DEV_NOTE :: operator+= is evaluated using operator+, so cant implicitly implement operator+=
        //DEV_NOTE :: operator+ is implemented this way in C#
        /**
        * Returns the value of the given vector added to this.
        */
        public static MyVector3 operator +( MyVector3 lv, MyVector3 rv)
        {
            lv.x += rv.x;
            lv.y += rv.y;
            lv.z += rv.z;
            return lv;
        }

        //DEV_NOTE :: operator-= is evaluated using operator- so cant implicitly implement operator-=
        //DEV_NOTE :: operator- is implemented this way in C#
        /**
        * Returns the value of the given vector subtracted to this.
        */
        public static MyVector3 operator -(MyVector3 lv, MyVector3 rv)
        {
            lv.x -= rv.x;
            lv.y -= rv.y;
            lv.z -= rv.z;
            return lv;
        }

        //DEV_NOTE :: operator*= is evaluated using operator* so cant implicitly implement operator*=
        //DEV_NOTE :: operator* is implemented this way in C#
        /**
        * Returns the value of the given vector added to this.
        */
        public static MyVector3 operator *(MyVector3 lv, double value)
        {
            lv.x *= value;
            lv.y *= value;
            lv.z *= value;
            return lv;
        }

        /**
         * Calculates and returns a component-wise product of this
         * vector with the given vector.
         */
        public MyVector3 ComponentProduct(MyVector3 vector)
        {
            return new MyVector3(x* vector.x, y* vector.y, z* vector.z);
        }

        /**
         * Performs a component-wise product with the given vector and
         * sets this vector to its result.
         */
        public void ComponentProductUpdate( MyVector3 vector)
        {
            x *= vector.x;
            y *= vector.y;
            z *= vector.z;
        }

        /**
         * Calculates and returns the vector product of this vector
         * with the given vector.
         */
        public MyVector3 VectorProduct( MyVector3 vector)
        {
            return new MyVector3(y* vector.z-z* vector.y,
                           z* vector.x-x* vector.z,
                           x* vector.y-y* vector.x);
        }

        //DEV_NOTE :: operator%= is evaluated using operator% so cant implicitly implement operator%=
        //DEV_NOTE :: operator% is implemented this way in C#
        /**
         * Calculates and returns the vector product of this vector
         * with the given vector.
         */
        public static MyVector3 operator %(MyVector3 lv,  MyVector3 vector)
        {
            lv.x = lv.y * vector.z - lv.z * vector.y;
            lv.y = lv.z * vector.x - lv.x * vector.z;
            lv.z = lv.x * vector.y - lv.y * vector.x;
            return lv;
        }

        /**
         * Calculates and returns the scalar product of this vector
         * with the given vector.
         */
        public double ScalarProduct( MyVector3 vector) 
        {
            return x* vector.x + y* vector.y + z* vector.z;
        }

        /**
         * Calculates and returns the scalar product of this vector
         * with the given vector.
         */
        public static double operator *(MyVector3 lv, MyVector3 vector) 
        {
            return lv.x* vector.x + lv.y * vector.y + lv.z * vector.z;
        }

        /**
         * Adds the given vector to this, scaled by the given amount.
         */
        public void AddScaledVector(MyVector3 vector, double scale)
        {
            x += vector.x* scale;
            y += vector.y* scale;
            z += vector.z* scale;
        }

        /** Gets the magnitude of this vector. */
        public double Magnitude()
        {
            return System.Math.Sqrt(x * x + y * y + z * z);
        }

        /** Gets the squared magnitude of this vector. */
        public double SquareMagnitude()
        {
            return x* x+y* y+z* z;
        }

        /** Limits the size of the vector to the given maximum. */
        public void Trim(double size)
        {
            if (SquareMagnitude() > size * size)
            {
                Normalise();
                x *= size;
                y *= size;
                z *= size;
            }
        }

        //DEV_NOTE :: can't assign using this keyword(at least in classes) in C#
        /** Turns a non-zero vector into a vector of unit length. */
        public void Normalise()
        {
            double l = Magnitude();
            if (l > 0)
            {
                l = ((double)1) / l;//get the inverse of l
                this.x *= l;
                this.y *= l;
                this.z *= l;
            }
        }

        /** Returns the normalised version of a vector. */
        public MyVector3 Unit() 
        {
            MyVector3 result = this;
            result.Normalise();
            return result;
        }

        //DEV_NOTE operator== is implemented this way in C#
        /** Checks if the two vectors have identical components. */
        public static bool operator ==(MyVector3 lv ,MyVector3 other) 
        {
            return lv.x == other.x &&
                lv.y == other.y &&
                lv.z == other.z;
        }

        /** Checks if the two vectors have non-identical components. */
        public static bool operator !=(MyVector3 lv, MyVector3 other)
        {
            return !(lv == other);
        }

        //DEV_NOTE :: operator< is implemented this way in C#
        /**
         * Checks if this vector is component-by-component less than
         * the other.
         *
         * @note This does not behave like a single-value comparison:
         * !(a<b) does not imply (b >= a).
         */
        public static bool operator <(MyVector3 lv, MyVector3 other) 
        {
            return lv.x < other.x && lv.y <other.y && lv.z <other.z;
        }

        //DEV_NOTE :: operator> is implemented this way in C#
        /**
         * Checks if this vector is component-by-component less than
         * the other.
         *
         * @note This does not behave like a single-value comparison:
         * !(a < b) does not imply (b >= a).
         */
        public static bool operator >(MyVector3 lv, MyVector3 other)
        {
            return lv.x > other.x && lv.y > other.y && lv.z > other.z;
        }

        //DEV_NOTE :: operator<= is implemented this way in C#
        /**
         * Checks if this vector is component-by-component less than
         * the other.
         *
         * @note This does not behave like a single-value comparison:
         * !(a <= b) does not imply (b > a).
         */
        public static bool operator <=(MyVector3 lv, MyVector3 other)
        {
            return lv.x <= other.x && lv.y <= other.y && lv.z <= other.z;
        }

        //DEV_NOTE :: operator>= is implemented this way in C#
        /**
         * Checks if this vector is component-by-component less than
         * the other.
         *
         * @note This does not behave like a single-value comparison:
         * !(a <= b) does not imply (b > a).
         */
        public static bool operator >=(MyVector3 lv, MyVector3 other)
        {
            return lv.x >= other.x && lv.y >= other.y && lv.z >= other.z;
        }

        /** Zero all the components of the vector. */
        public void Clear()
        {
            x = y = z = 0;
        }

        /** Flips all the components of the vector. */
        public void Invert()
        {
            x = -x;
            y = -y;
            z = -z;
        }


        //-------------------------------finished original vector3 class
        public override bool Equals(object obj)
        {
            var vector = obj as MyVector3;
            return vector != null &&
                   x == vector.x &&
                   y == vector.y &&
                   z == vector.z;
        }

        public override int GetHashCode()
        {
            var hashCode = 373119288;
            hashCode = hashCode * -1521134295 + x.GetHashCode();
            hashCode = hashCode * -1521134295 + y.GetHashCode();
            hashCode = hashCode * -1521134295 + z.GetHashCode();
            return hashCode;
        }

        
        public MyVector3(MyVector3 v)
        {

            x = v.x;
            y = v.y;
            z = v.z;
        }

        //DEV_NOTE :: operator*= is evaluated using operator* so cant implicitly implement operator*=
        //DEV_NOTE :: operator* is implemented this way in C#
        /**
        * Returns the value of the given vector added to this.
        */
        public static MyVector3 operator *(double value, MyVector3 lv)
        {
            lv.x *= value;
            lv.y *= value;
            lv.z *= value;
            return lv;
        }


        //finished original vector3 class


        /// <summary>
        /// converts vector3 to MyVector3
        /// </summary>
        /// <param name="v">vector3 to convert to MyVector3</param>
        /// <returns></returns>
        public static MyVector3 ConvertToMyVector3(Vector3 v)
        {
            return new MyVector3(v.x, v.y, v.z);
        }

        /// <summary>
        /// converts vector3 to MyVector3
        /// </summary>
        /// <param name="v">vector3 to convert to MyVector3</param>
        /// <returns></returns>
        public static Vector3 ConvertToVector3(MyVector3 v)
        {
            return new Vector3((float)v.x, (float)v.y, (float)v.z);
        }

       
        /** Turns a non-zero vector into a vector of unit length. */
        public MyVector3 Normalized()
        {
            MyVector3 normalized = new MyVector3(this);
            double l = Magnitude();
            if (l > 0)
            {
                l = ((double)1) / l;//get the inverse of l
                normalized.x *= l;
                normalized.y *= l;
                normalized.z *= l;
            }
            return normalized;
        }

        /**
        * Returns the value of the given vector added to this.
        */
        public static MyVector3 operator /(MyVector3 lv, double value)
        {
            if (value == 0)
            {
                Debug.LogError("MyVector3 divison by zero");
                GameManager.Quit();
            }
            value = 1 / value;

            lv.x *= value;
            lv.y *= value;
            lv.z *= value;
            return lv;
        }
    }
}


