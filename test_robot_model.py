import tempfile
from pathlib import Path
import struct
import unittest
from robot_model import parse_model,stl_vertices

class ModelTests(unittest.TestCase):
    def test_binary_mesh_origin_scale_and_joint_units_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'mesh.stl').write_bytes(b'\0'*80+struct.pack('<I',1)+struct.pack('<12fH',0,0,1,0,0,0,1,0,0,0,1,0,0))
            urdf=p/'robot.urdf';urdf.write_text('''<robot name="test"><link name="base_link_arm"/><link name="tip"><visual><origin xyz="1 2 3" rpy="0 0 1"/><geometry><mesh filename="package://test/mesh.stl" scale="0.001 0.001 0.001"/></geometry></visual></link><joint name="joint1" type="revolute"><parent link="base_link_arm"/><child link="tip"/><origin xyz="0 0 1"/><axis xyz="0 0 1"/><limit lower="-1" upper="2"/></joint></robot>''')
            m=parse_model(urdf,lambda _:tmp);v=m['links'][1]['visuals'][0]
            self.assertEqual(v['vertices'],[0,0,0,1,0,0,0,1,0])
            self.assertEqual(v['scale'],[.001]*3);self.assertEqual(v['xyz'],[1,2,3])
            self.assertEqual(m['joints'][0]['upper'],2);self.assertEqual(m['joints'][0]['axis'],[0,0,1])
            urdf.write_text(urdf.read_text().replace('test/mesh.stl','test/../outside.stl'))
            with self.assertRaisesRegex(ValueError,'escapes'):parse_model(urdf,lambda _:tmp)

if __name__=='__main__':unittest.main()
