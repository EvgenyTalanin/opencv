#include "test_precomp.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/scenetext.hpp"

using namespace cv;
using namespace std;

TEST(Contrib_Matas, regression)
{
    string imgName = "/home/evgeny/argus/opencv_fork/opencv/samples/cpp/matas/incorrect640.jpg";
    //imgName = "/home/evgeny/argus/opencv_fork/opencv/samples/cpp/matas/ontario_small.jpg";
    Mat im = imread(imgName);

    set<Region, RegionComp> t;
    set<Region, RegionComp> m;

    SceneTextLocalizer scl(im, 100);
    t = scl.GroundTruth();
    m = scl.MatasLike();

    EXPECT_EQ(t.size(), m.size());

    for(set<Region, RegionComp>::iterator itt = t.begin(), itm = m.begin(); itt != t.end(); ++itt, ++itm)
    {
        Region rt = *itt;
        Region rm = *itm;

        EXPECT_EQ(rt.Bounds(), rm.Bounds());
        EXPECT_EQ(rt.Area(), rm.Area());
        EXPECT_EQ(rt.Perimeter(), rm.Perimeter());
        EXPECT_EQ(rt.Euler(), rm.Euler());
        EXPECT_EQ(rt.CrossingsCount(), rm.CrossingsCount());

        for(int i = rt.Bounds().y; i < rt.Bounds().y + rt.Bounds().height; i++)
        {
            EXPECT_EQ(rt.Crossings(i), rm.Crossings(i));
        }
    }
}
