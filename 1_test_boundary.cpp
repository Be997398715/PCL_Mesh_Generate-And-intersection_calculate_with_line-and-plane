// triangulation_test.cpp: 定义控制台应用程序的入口点。
//
#include<iostream>
#include<pcl/point_types.h>
#include<pcl/io/pcd_io.h>
#include<pcl/kdtree/kdtree_flann.h>
#include<pcl/features/normal_3d.h>
#include<pcl/surface/gp3.h>  //贪婪投影三角化算法
#include<pcl/visualization/pcl_visualizer.h>
#include<boost/math/special_functions/round.hpp>
#include <pcl/io/vtk_io.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
using namespace pcl;
using namespace std;

struct Point3f{
    double x;
    double y;
    double z;
};

void writeOBJ(pcl::PolygonMesh& mesh, string filename);
void readOBJ( string filename, pcl::PolygonMesh& mesh, pcl::PointCloud<pcl::PointXYZ> & tempCloud);
void Calculate_intersection(
double* newJiaoPoint,
double Lx1, double Ly1, double Lz1, double Lx2, double Ly2, double Lz2,
double Px1, double Py1, double Pz1, double Px2, double Py2, double Pz2,
double Px3, double Py3, double Pz3);

/*
Author @ 997398715@qq.com
Module Explain:
    1> Calculate the intersection of the line and mesh in the space.
    2> Generate mesh from cloudpoints.
*/
void visualization_intersection(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_boundary) {
  	//pcl::visualization::PCLVisualizer viewer1("viewer1");
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    //viewer1.addPointCloud<pcl::PointXYZ> (tempCloud.makeShared(), "test");
	//viewer1.spin();
    pcl::visualization::PCLVisualizer viewer1("viewer_intersection");

    // 添加需要显示的点云数据
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
    viewer1.addPointCloud<pcl::PointXYZ>(cloud, single_color, "viewer_intersection1");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color_(cloud_boundary, 255, 0, 0);
    viewer1.addPointCloud<pcl::PointXYZ>(cloud_boundary, single_color_, "viewer_intersection2");
    viewer1.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "viewer_intersection2");
    viewer1.spin();
}


int main()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::PCLPointCloud2 cloud_blob;
	//pcl::io::loadPCDFile("../rabbit.pcd", cloud_blob);
	//pcl::fromPCLPointCloud2(cloud_blob, *cloud);
	string file_path = "../input.txt";
    std::ifstream file(file_path.c_str());//c_str()：生成一个const char*指针，指向以空字符终止的数组。
    std::string line;
    pcl::PointXYZ point_;
    while (getline(file, line)) {
        std::stringstream ss(line);
        ss >> point_.x;
        ss >> point_.y;
        ss >> point_.z;
        cloud->push_back(point_);
    }
    file.close();


	//Normal estimation
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal>n;  //法线估计对象
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>); //存储法线的向量
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(20);
	n.compute(*normals);  //估计法线存储位置

	//Concatenate the XYZ and normal field
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);  //连接字段
	//point_with_normals = cloud + normals

	//定义搜索树对象
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud(cloud_with_normals); //点云搜索树

	//Initialize objects
	pcl::GreedyProjectionTriangulation<pcl::PointNormal>gp3;  //定义三角化对象
	pcl::PolygonMesh triangles; //定义最终三角化的网络模型

	gp3.setSearchRadius(0.75);  //设置连接点之间的最大距离（即为三角形的最大边长）

	//设置各参数值
	gp3.setMu(2.5);    //设置被样本点搜索其最近邻点的最远距离，为了使用点云密度的变化
	gp3.setMaximumNearestNeighbors(100); //样本点可搜索的领域个数
	gp3.setMaximumSurfaceAngle(M_PI / 4);  //某点法向量方向偏离样本点法线的最大角度45°
	gp3.setMinimumAngle(M_PI / 18);  //设置三角化后得到的三角形内角最小角度为10°
	gp3.setMaximumAngle(2 * M_PI / 3); //设置三角化后得到的三角形内角的最大角度为120°
	gp3.setNormalConsistency(false); //设置该参数保证法线朝向一致

	//Get Result
	gp3.setInputCloud(cloud_with_normals);  //设置输入点云为有向点云
	gp3.setSearchMethod(tree2); //设置搜索方式
	gp3.reconstruct(triangles); //重建提取三角化

	//附加顶点信息
	vector<int>parts = gp3.getPartIDs();
	vector<int>states = gp3.getPointStates();

	//Viewer
	pcl::visualization::PCLVisualizer viewer("viewer");
	viewer.addPolygonMesh(triangles);
	viewer.spin();

	//save mesh
    pcl::io::saveVTKFile ("../mesh.vtk", triangles);
    string s = "../mesh.obj";
    writeOBJ(triangles, s);
    cout << triangles.polygons.size() << endl;


    //read and calculate
    pcl::PolygonMesh mesh;
    pcl::PointCloud<pcl::PointXYZ> tempCloud;
    pcl::PointCloud<pcl::PointXYZ> intersection_cloud;
    pcl::PointXYZ point;

    readOBJ(s, mesh, tempCloud);
    /*
    reference:
    https://blog.csdn.net/liangsongjun/article/details/82021777?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&dist_request_id=ba215bec-c622-4d00-8da0-5f6e398550d3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control
    */
    double Lx1,Ly1,Lz1,Lx2,Ly2,Lz2,Px1,Py1,Pz1,Px2,Py2,Pz2,Px3,Py3,Pz3;
    double* jiaopoint = new double[3];
    Lx1=0; Ly1=0; Lz1=0;    //ine point 1
    Lx2=0; Ly2=0; Lz2=1;   //line point 2

    int p1_,p2_,p3_;
    int numf = mesh.polygons.size(), numv = tempCloud.size();
    vector<int> status;
    vector<Point3f> jiaodian_all;
    Point3f jiaodian;
    int cnt = 0;
	for (int i = 0; i < numf; i++){
		p1_ = mesh.polygons[i].vertices[0];
		p2_ = mesh.polygons[i].vertices[1];
		p3_ = mesh.polygons[i].vertices[2];

		Px1 = tempCloud.points[p1_].x;
		Py1 = tempCloud.points[p1_].y;
		Pz1 = tempCloud.points[p1_].z;

		Px2 = tempCloud.points[p2_].x;
		Py2 = tempCloud.points[p2_].y;
		Pz2 = tempCloud.points[p2_].z;

		Px3 = tempCloud.points[p3_].x;
		Py3 = tempCloud.points[p3_].y;
		Pz3 = tempCloud.points[p3_].z;

		Calculate_intersection(
            jiaopoint,
            Lx1, Ly1, Lz1,
            Lx2, Ly2, Lz2,
            Px1, Py1, Pz1,
            Px2, Py2, Pz2,
            Px3, Py3, Pz3
		);
		if(jiaopoint==NULL){
		    status.push_back(0);
		    jiaodian.x = 0;
		    jiaodian.y = 0;
		    jiaodian.z = 0;
		    jiaodian_all.push_back(jiaodian);
		}else{
		    cnt = cnt + 1;
		    status.push_back(1);
		    jiaodian.x = jiaopoint[0];
		    jiaodian.y = jiaopoint[1];
		    jiaodian.z = jiaopoint[2];
		    jiaodian_all.push_back(jiaodian);

		    point.x = jiaopoint[0];
            point.y = jiaopoint[1];
            point.z = jiaopoint[2];
            intersection_cloud.points.push_back(point);
		    cout << "jiaodian :" << jiaodian.x << " " << jiaodian.y << " " << jiaodian.z << endl;
		}
	}
    cout << "all jiaodian size :" << cnt << endl;
    visualization_intersection(tempCloud.makeShared(), intersection_cloud.makeShared() );

    return 0;
}

/*
直线上两点坐标A(Lx1, Ly1, Lz1),B(Lx2, Ly2, Lz2)，
与平面上的三点坐标C(Px1, Py1, Pz1),D(Px2, Py2, Pz2),E(Px3, Py3, Pz3)
*/
void Calculate_intersection(
double* newJiaoPoint,
double Lx1, double Ly1, double Lz1, double Lx2, double Ly2, double Lz2,
double Px1, double Py1, double Pz1, double Px2, double Py2, double Pz2,
double Px3, double Py3, double Pz3)
{
    //double* newJiaoPoint = new double[3];
    //L直线矢量
    double m = Lx2 - Lx1;
    double n = Ly2 - Ly1;
    double p = Lz2 - Lz1;
    //MessageBox.Show(m.ToString("#0.#") + "," + n.ToString("#0.#") + "," + p.ToString("#0.#,"));

    //平面方程Ax+BY+CZ+d=0 行列式计算
    double A = Py1 * Pz2 + Py2 * Pz3 + Py3 * Pz1 - Py1 * Pz3 - Py2 * Pz1 - Py3 * Pz2;
    double B = Px1 * Pz2 + Px2 * Pz3 + Px3 * Pz1 - Px3 * Pz2 - Px2 * Pz1 - Px1 * Pz3;
    double C = Px1 * Py2 + Px2 * Py3 + Px3 * Py1 - Px1 * Py3 - Px2 * Py1 - Px3 * Py2;
    double D = Px1 * Py2 * Pz3 + Px2 * Py3 * Pz1 + Px3 * Py1 * Pz2 - Px1 * Py3 * Pz2 - Px2 * Py1 * Pz3 - Px3 * Py2 * Pz1;
    //MessageBox.Show(A.ToString("#0.#") + "," + B.ToString("#0.#") + "," + C.ToString("#0.#,") + "," + D.ToString("#0.#,"));
    //系数比值 t=-(Axp+Byp+Cxp+D)/(A*m+B*n+C*p)

    if (A*m+B*n+C*p == 0)  //判断直线是否与平面平行
    {
        newJiaoPoint = NULL;
    }
    else
    {
        double t = (Lx1 * A + Ly1 * B + Lz1 * C + D) / (A * m + B * n + C * p);
        newJiaoPoint[0] = Lx1 + m * t;
        newJiaoPoint[1] = Ly1 + n * t;
        newJiaoPoint[2] = Lz1 + p * t;
    }
}

void writeOBJ(pcl::PolygonMesh& mesh, string filename){
	FILE* f;
	f = fopen(filename.c_str(), "w");
    pcl::PointCloud<pcl::PointXYZ> tempCloud;
    pcl::fromPCLPointCloud2(mesh.cloud,tempCloud);//polygonMesh中的cloud是PointCloud2类型的无法直接读取，所以这里进行转换
	int numf = mesh.polygons.size(), numv = tempCloud.size();
	fprintf(f, "# faces %d, vetices %d \n", numf, numv);

	for (int i = 0; i < numv; i++){
	try{
		fprintf(f, "v %f %f %f\n", tempCloud.points[i].x, tempCloud.points[i].y, tempCloud.points[i].z);
	}
	catch(...){ ; }
	}

	for (int i = 0; i < numf; i++){
		fprintf(f, "f %d %d %d\n", mesh.polygons[i].vertices[0] + 1,mesh.polygons[i].vertices[1]+ 1, mesh.polygons[i].vertices[2] + 1);
	}
	fclose(f);
    return;
}


void readOBJ( string filename, pcl::PolygonMesh& mesh, pcl::PointCloud<pcl::PointXYZ> & tempCloud){
	FILE* f;
	f = fopen(filename.c_str(), "r");
	//pcl::PointCloud<pcl::PointXYZ> tempCloud;
	char buf[256];
	while (1)
	{
		if (fgets(buf, 256, f) == NULL) break;
		if (buf[0] == '\n' | buf[0] == '#') continue;
		char* p;
		p = buf;
		if (p[0] == 'v'){
			pcl::PointXYZ tp;
			sscanf(p, "v%f%f%f", &tp.x, &tp.y, &tp.z);
			tempCloud.points.push_back(tp);
		}
		else{
			if (p[0] == 'f'){
				pcl::Vertices ttri;
				ttri.vertices.resize(3);
				sscanf(p, "f%d%d%d", &ttri.vertices[0], &ttri.vertices[1], &ttri.vertices[2]);//obj文件中下标从1开始，c++中的数组或者向量下标从0
				//Triangle ttri0;
				//sscanf(p, "f%d/%d/%d/%d/%d/%d/", &ttri.P0Index, &ttri0.P0Index, &ttri.P1Index, &ttri0.P1Index, &ttri.P2Index, &ttri0.P2Index);
				ttri.vertices[0] -= 1, ttri.vertices[1] -= 1, ttri.vertices[2] -= 1;
				mesh.polygons.push_back(ttri);
			}
		}

	}
	pcl::toPCLPointCloud2(tempCloud, mesh.cloud);
    return;
}