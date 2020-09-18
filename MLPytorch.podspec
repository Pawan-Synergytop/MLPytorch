Pod::Spec.new do |s|

# 1
s.platform = :ios
s.ios.deployment_target = '13.0'
s.name = "MLPytorch"
s.summary = "MLPytorch lets a user select an ice cream flavor."
s.requires_arc = true

# 2
s.version = "0.1.0"

# 3
s.license = { :type => "MIT", :file => "LICENSE" }

# 4 - Replace with your name and e-mail address
s.author = { "Pawan Malviya" => "pawan.synergytop@gmail.com" }

# 5 - Replace this URL with your own GitHub page's URL (from the address bar)
s.homepage = "https://github.com/Pawan-Synergytop/MLPytorch"

# 6 - Replace this URL with your own Git URL from "Quick Setup"
s.source = { :git => "https://github.com/Pawan-Synergytop/MLPytorch.git",
             :tag => "#{s.version}" }

# 7
s.framework = "UIKit"
s.dependency 'LibTorch', '~>1.5.0'

# 8
s.source_files = "MLPytorch/**/*.{swift,mm,h}"

# 9
s.resources = "MLPytorch/**/*.{png,jpeg,jpg,xcassets}"

# 10
s.swift_version = "5"

end
