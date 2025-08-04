# Example of what might come in from "aruco"
data = "ID:4,9.0,2.0,3.4,1.3,0.3,0.4 ID:2,9.0,2.0,3.4,1.3,0.3,0.4 ID:5,9.0,2.0,3.4,1.3,0.3,0.4"
# Split tag records at "ID:"
datasp = data.split("ID:")
# Split each of those at comma
tagsp = [tag.split(",") for tag in datasp[1:]]    
# turn each of those little strings into a Float
tagspnum = [[float(x) for x in tag] for tag in tagsp]
# for each tag, turn the first number into an Int
for ii in range(len(tagspnum)):
    tagspnum[ii][0] = int(tagspnum[ii][0])
