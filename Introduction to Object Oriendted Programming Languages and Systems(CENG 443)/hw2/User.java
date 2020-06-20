import java.lang.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class User implements Runnable
{
	private final String name;
	private final List<String> wantedSeats;
	private ArrayList<ArrayList<Seat>> seats;
	private int seatColumnCount;
	
	public User(List<String> nameAndWantedSeats, ArrayList<ArrayList<Seat>> seats, int seatColumnCount)
	{
		this.name = nameAndWantedSeats.get(0);
		this.wantedSeats = nameAndWantedSeats.subList(1, nameAndWantedSeats.size());
		this.seats = seats;
		this.seatColumnCount = seatColumnCount;
	}
	
	private void UnlockPreviousLocks(int maxIndex)
	{
		try
		{
			Thread.sleep(50);//sleep 50 seconds to let the trylock statements fully execute
		}
		catch (InterruptedException e)
		{
			//System.out.println("thread sleep error, " + e);
		}
		for (int j = 0; j < maxIndex + 1; j++)//unlock all the previous seats
		{
			String seatID = wantedSeats.get(j);
			int seatIndex1 = (seatID.charAt(0) - 'A');
			int seatIndex2 = (seatID.charAt(1) - '0');
			seats.get(seatIndex1).get(seatIndex2).Unlock();
		}
	}
	
	private boolean TryLockSeats()
	{
		for (int i = 0; i < wantedSeats.size(); i++)
		{
			String seatID = wantedSeats.get(i);
			int seatIndex1 = (seatID.charAt(0) - 'A');
			int seatIndex2 = (seatID.charAt(1) - '0');
			
			//System.out.println("seatID" + seatID + " seatIndex<-"+seatIndex);
			
			switch (seats.get(seatIndex1).get(seatIndex2).CheckIsReserved())
			{
				case Reserved:
					// abort reserving of all seats
					UnlockPreviousLocks(i);
					Logger.LogFailedReservation(this.name, Arrays.deepToString(wantedSeats.toArray()), System.nanoTime(), "seats are not empty");
					return false;
				case NotReserved://no need to do anything. keeps the lock on for 50ms
					break;
				case Unknown:
					//since we couldn't acquire the lock, unlock all previous locks and try again
					UnlockPreviousLocks(i);
					return TryLockSeats();
			}
		}
		return true;
	}
	
	private void TryReserveSeats()
	{
		try
		{
			if (TryLockSeats() == false)//we want a seat that was previously reserved, so we cant reserve
			{
				return;
			}
			else
			{
				//try to reserve all the seats we want, unlock the seats if we encounter a database error
				if (Math.random() < 0.1)//database fail
				{
					UnlockPreviousLocks(wantedSeats.size() -1);
					
					Logger.LogDatabaseFailiure(this.name, Arrays.deepToString(wantedSeats.toArray()), System.nanoTime(), "database failed");
					Thread.sleep(100);
					//retry reserving
					TryReserveSeats();
				}
				else
				{
					for (int i = 0; i < wantedSeats.size(); i++) // update seat information -- seat.Reserve() automatically unlocks the seat so no problem
					{
						String seatID = wantedSeats.get(i);
						int seatIndex1 = (seatID.charAt(0) - 'A');
						int seatIndex2 = (seatID.charAt(1) - '0');
						seats.get(seatIndex1).get(seatIndex2).Reserve(this.name);//locks are unlocked in this function
					}
					Thread.sleep(50);
					Logger.LogSuccessfulReservation(this.name, Arrays.deepToString(wantedSeats.toArray()), System.nanoTime(), "Successful reservation");
				}
			}
		}
		catch (Exception e)
		{
			//System.out.println("Exception" + e + "is caught in user thread");
		}
	}
	
	@Override
	public void run()
	{
		TryReserveSeats();
	}
}