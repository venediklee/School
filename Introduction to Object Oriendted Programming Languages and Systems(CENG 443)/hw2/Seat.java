import java.lang.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;

public class Seat
{
	final public String seatName;
	private ReserveStatus isReserved;
	
	private ReentrantLock lock;
	
	public String NameOfReserver;
	
	
	public Seat(String seatName)
	{
		this.seatName = seatName;
		isReserved = ReserveStatus.NotReserved;
		lock = new ReentrantLock();
	}
	
	public ReserveStatus CheckIsReserved()//called before trying to reserve the seat to see if it is reserved or not
	{
		try
		{
			boolean gotLock = lock.tryLock((long) 50.0, TimeUnit.MILLISECONDS);
			if (gotLock)
			{
				return isReserved;
			}
			else
			{
				return ReserveStatus.Unknown;
			}
		}
		catch (InterruptedException e)
		{
			//System.out.println("tried locking isReserved for 50ms, couldn't do it");
			return ReserveStatus.Unknown;
		}
	}
	
	public void Unlock()//called if the seat is reserved when checking reserved status
	{
		try
		{
			lock.unlock();
		}
		catch (Exception e)
		{
			//System.out.println("trying to unlock a lock we dont own failed");
		}
	}
	
	public void Reserve(String userName)//called after ALL the wanted seats are not reserved
	{
		//System.out.println(userName + " reserving " + this.seatName + this.isReserved);
		this.isReserved = ReserveStatus.Reserved;
		Unlock();
		NameOfReserver = userName;
	}
	
}

