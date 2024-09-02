import Image from "next/image";
import { Button } from "@/components/ui/button"
import {UserButton} from "@clerk/nextjs"

export default function SetupPage() {
  return (
   
    <div className="p-4 flex items-centre justify-center text-lg">
      <Button className="bg-black text-white hover:bg-white hover:text-black" >Manage Shop</Button>
      </div>
 
  );
}
