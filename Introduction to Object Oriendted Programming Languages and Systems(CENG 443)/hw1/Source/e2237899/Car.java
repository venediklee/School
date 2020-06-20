public class Car
{
	private int carNo;
	private String driverName;
	private double totalTime;
	private Tire tire;
	
	public Car()
	{
	}
	
	public Car(String driverName, int carNo, Tire tire)
	{
		// Fill this method
		this.driverName = driverName;
		this.carNo = carNo;
		this.tire = tire;
		this.totalTime = 0;
	}
	
	public Tire getTire()
	{
		return tire;
	}
	
	public void setTire(Tire tire)
	{
		this.tire = tire;
	}
	
	public String getDriverName()
	{
		return driverName;
	}
	
	public void setDriverName(String driverName)
	{
		this.driverName = driverName;
	}
	
	public int getCarNo()
	{
		return carNo;
	}
	
	public void setCarNo(int carNo)
	{
		this.carNo = carNo;
	}
	
	public double getTotalTime()
	{
		return totalTime;
	}
	
	public void tick(TrackFeature feature)
	{
		totalTime += feature.getDistance() / this.getTire().getSpeed() + Math.random();
	}
	
	public int sortByTimingAscending(Car another)
	{
		if (this.getTotalTime() < another.getTotalTime())
		{
			return -1;
		}
		else
		{
			return 1;
		}
	}
	
	public void pitStop()
	{
		this.totalTime += 25;
		this.setTire(this.getTire().getPitStopTire());
	}
	
}
