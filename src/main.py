import argparse
from StreamDMM import StreamGL, load_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Dataset name")
    parser.add_argument("--lc", type=int, default=10, help="Length of the current window")
    parser.add_argument("--dt", type=int, default=1, help="Time step")
    parser.add_argument("--start_date", type=str, default="2010-01-01", help="Start date")
    parser.add_argument("--end_date", type=str, default="2022-12-31", help="End date")
    args = parser.parse_args()

    path = "../data/" + args.data + ".csv.gz"

    X = load_tensor(path, 
                    time_key="date", modes=["query", "geo"], values="volume", 
                    sampling_rate="W", start_date=args.start_date, end_date=args.end_date, 
                    scaler="each", verbose=True)
    
    streamGL = StreamGL()
    streamGL.run(X, lc=args.lc, dt=args.dt)

if __name__ == "__main__":
    main()