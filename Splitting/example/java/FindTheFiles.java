public class FindTheFiles {
	public void FindTheFilesInPath(String path) {
        File dir = new File(path);
        String[] children = dir.list();
        if (children == null) {
           System.out.println("No such dir");
		}
        else {
			 for (int i = 0; i < children.length; i++) {
                String filename = children[i];
                System.out.println(filename);   
				}
            }
        }
    }
