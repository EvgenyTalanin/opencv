#include "test_precomp.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/scenetext.hpp"

using namespace cv;
using namespace std;

TEST(Contrib_Matas, regression)
{
    string imgName = "../opencv/samples/cpp/matas/incorrect640.jpg";
    Mat im = imread(imgName);

    set<Region, RegionComp> t;
    set<Region, RegionComp> m;

    SceneTextLocalizer scl(im, 100);
    t = scl.GroundTruth();
    m = scl.MatasLike();

    EXPECT_TRUE(t.size() == m.size());

    for(set<Region, RegionComp>::iterator itt = t.begin(), itm = m.begin(); itt != t.end(); ++itt, ++itm)
    {
        Region rt = *itt;
        Region rm = *itm;

        EXPECT_TRUE(rt.Bounds() == rm.Bounds());
        EXPECT_TRUE(rt.Area() == rm.Area());
        EXPECT_TRUE(rt.Perimeter() == rm.Perimeter());
        EXPECT_TRUE(rt.Euler() == rm.Euler());
        EXPECT_TRUE(rt.CrossingsCount() == rm.CrossingsCount());

        for(int i = rt.Bounds().y; i < rt.Bounds().y + rt.Bounds().height; i++)
        {
            EXPECT_TRUE(rt.Crossings(i) == rm.Crossings(i));
        }
    }
}
