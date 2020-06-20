import java.util.ArrayList;
import java.util.Collections;

public class Session
{
	
	private Track track;
	private ArrayList<Team> teamList;
	private int totalLaps;
	
	public Session()
	{
	}
	
	public Session(Track track, ArrayList<Team> teamList, int totalLaps)
	{
		this.track = track;
		this.teamList = teamList;
		this.totalLaps = totalLaps;
	}
	
	public Track getTrack()
	{
		return track;
	}
	
	public void setTrack(Track track)
	{
		this.track = track;
	}
	
	public ArrayList<Team> getTeamList()
	{
		return teamList;
	}
	
	public void setTeamList(ArrayList<Team> teamList)
	{
		this.teamList = teamList;
	}
	
	public int getTotalLaps()
	{
		return totalLaps;
	}
	
	public void setTotalLaps(int totalLaps)
	{
		this.totalLaps = totalLaps;
	}
	
	public void simulate()
	{
		if (!track.isValidTrack())
		{
			System.out.println("Track is invalid.Simulation aborted!");
			return;
		}
		
		System.out.println("Track is valid.Strating simulation on " + track.getTrackName() +
								   " for " + totalLaps + " laps.");
		
		
		for (int lap = 0; lap < totalLaps; lap++)
		{
			for (TrackFeature feature : track.getFeatureList())
			{
				for (Team team : teamList)
				{
					//int carCount = team.getCarList().size();
					for (Car car : team.getCarList())
					{
						
						car.tick(feature);
						car.getTire().tick(feature);
						if (car.getTire().getDegradation() > 70) // pit stop
						{
							car.pitStop();
						}
					}
				}
			}
		}
		
		printWinnerTeam();
		printTimingTable();
	}
	
	public String printWinnerTeam()
	{
		//Fill this method
		int winnerTeamIndex = 0;
		int winnerCarInTeamIndex = 0;
		
		int contestantTeamIndex = 0;
		for (Team team : teamList)
		{
			int contestantCarIndex = 0;
			for (Car contestantCar : team.getCarList())
			{
				if (contestantCar.getTotalTime() <
						teamList.get(winnerTeamIndex).getCarList().get(winnerCarInTeamIndex).getTotalTime())
				{
					winnerTeamIndex = contestantTeamIndex;
					winnerCarInTeamIndex = contestantCarIndex;
				}
				contestantCarIndex++;
			}
			contestantTeamIndex++;
		}
		
		StringBuilder result = new StringBuilder();
		
		result.append("Team " + teamList.get(winnerTeamIndex).getName() + " wins.");
		result.append(teamList.get(winnerTeamIndex).getTeamColors()[0]);
		
		int teamColorCount = teamList.get(winnerTeamIndex).getTeamColors().length;
		
		for (int i = 1; i < teamColorCount - 1; i++)//last color gets and keyword
		{
			result.append(", " + teamList.get(winnerTeamIndex).getTeamColors()[i]);
		}
		
		if (teamColorCount > 1)
		{
			result.append(" and " + teamList.get(winnerTeamIndex).getTeamColors()[
					teamColorCount - 1]);
		}
		
		result.append(" flags are waving everywhere.");
		
		System.out.println(result);
		
		return result.toString();
	}
	
	private String printTimingTable()
	{
		// 59 check the sort thingy Fill this method
		ArrayList<Car> cars = new ArrayList<Car>();
		
		for (Team team : teamList)
		{
			for (Car car : team.getCarList())
			{
				cars.add(car);
			}
		}
		
		cars.sort(Car::sortByTimingAscending);
		
		StringBuilder results = new StringBuilder();
		
		for (Car car : cars)
		{
			double time = car.getTotalTime();
			
			int hours = (int) (time / 3600);
			time -= hours * 3600;
			int minutes = (int) (time / 60);
			time -= minutes * 60;
			int seconds = (int) time;
			time -= seconds;
			
			results.append(car.getDriverName() + "(" + car.getCarNo() + "): " +
								   String.valueOf(hours) + ":" +
								   String.valueOf(minutes) + ":" +
								   String.valueOf(seconds) + "." +
								   String.valueOf((int) (time * 1000))
								   + "\n");
		}
		
		System.out.println(results.substring(0, results.length() - 1));
		
		return results.substring(0, results.length() - 1);
		
	}
}